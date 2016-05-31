import h5py
import theano.tensor as T
import os.path as osp

from layers import MLP


class SceneMlp(object):
    """
    multi-layer perceptron used to predict scene-specific context
    """
    def __init__(self, name='scene_mlp', layer_sizes=(2048, 1024, 1024, 80), model_file=None):
        self.name = name
        if model_file is not None:
            with h5py.File(model_file, 'r') as f:
                layer_sizes = f.attrs['layer_sizes']
        self.config = {'layer_sizes': layer_sizes}

        # define inputs
        x = T.matrix('x')
        y = T.matrix('y')
        self.inputs = [x, y]

        # define computation graph
        self.mlp = MLP(layer_sizes=layer_sizes, name='mlp', output_type='softmax')
        self.proba = self.mlp.compute(x)
        self.log_proba = T.log(self.proba)

        # define costs
        def kl_divergence(p, q):
            kl = T.mean(T.sum(p * T.log((p+1e-30)/(q+1e-30)), axis=1))
            kl += T.mean(T.sum(q * T.log((q+1e-30)/(p+1e-30)), axis=1))
            return kl
        kl = kl_divergence(self.proba, y)
        acc = T.mean(T.eq(self.proba.argmax(axis=1), y.argmax(axis=1)))
        self.costs = [kl, acc]

        # layers and parameters
        self.layers = [self.mlp]
        self.params = sum([l.params for l in self.layers], [])

        # load weights from file, if model_file is not None
        if model_file is not None:
            self.load_weights(model_file)

    def save_to_dir(self, save_dir, idx='0'):
        save_file = osp.join(save_dir, self.name+'.h5.' + str(idx))
        for l in self.layers:
            l.save_weights(save_file)
        with h5py.File(save_file) as f:
            for k, v in self.config.items():
                f.attrs[k] = v

    def load_weights(self, model_file):
        for l in self.layers:
            l.load_weights(model_file)
