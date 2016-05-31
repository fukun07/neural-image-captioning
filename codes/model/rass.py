import h5py
import theano
import theano.tensor as T
import os.path as osp
import numpy as np

from layers import split_state, MLP, BasicLSTM, Embedding, Attention


class Model(object):
    """
    region attention + scene-specific contexts
    """
    def __init__(self, name='rass', nimg=2048, nh=512, nw=512, na=512, nout=8843, ns=80, npatch=30, model_file=None):
        self.name = name
        if model_file is not None:
            with h5py.File(model_file, 'r') as f:
                nimg = f.attrs['nimg']
                nh = f.attrs['nh']
                nw = f.attrs['nw']
                na = f.attrs['na']
                ns = f.attrs['ns']
                nout = f.attrs['nout']
        self.config = {'nimg': nimg, 'nh': nh, 'nw': nw, 'na': na, 'nout': nout, 'ns': ns, 'npatch': npatch}

        # word embedding layer
        self.embedding = Embedding(n_emb=nout, dim_emb=nw, name=self.name+'@embedding')

        # initialization mlp layer
        self.init_mlp = MLP(layer_sizes=[na, 2*nh], output_type='tanh', name=self.name+'@init_mlp')
        self.proj_mlp = MLP(layer_sizes=[nimg, na], output_type='tanh', name=self.name+'@proj_mlp')

        # attention layer
        self.attention = Attention(dim_item=na, dim_context=na+nw+nh, hsize=nh, name=self.name+'@attention')

        # lstm
        self.lstm = BasicLSTM(dim_x=na+nw+ns, dim_h=nh, name=self.name+'@lstm')

        # prediction mlp
        self.pred_mlp = MLP(layer_sizes=[na+nh+nw+ns, nout], output_type='softmax', name=self.name+'@pred_mlp')

        # inputs
        cap = T.imatrix('cap')
        img = T.tensor3('img')
        scene = T.matrix('scene')
        self.inputs = [cap, img, scene]

        # go through sequence
        feat = self.proj_mlp.compute(img)
        init_e = feat.mean(axis=1)
        init_state = T.concatenate([init_e, self.init_mlp.compute(init_e)], axis=-1)
        (state, self.p, loss, self.alpha), _ = theano.scan(fn=self.scan_func,
                                                           sequences=[cap[0:-1, :], cap[1:, :]],
                                                           outputs_info=[init_state, None, None, None],
                                                           non_sequences=[feat, scene])

        # loss function
        loss = T.mean(loss)
        self.costs = [loss]

        # layers and parameters
        self.layers = [self.embedding, self.init_mlp, self.proj_mlp, self.attention, self.lstm, self.pred_mlp]
        self.params = sum([l.params for l in self.layers], [])

        # load weights from file, if model_file is not None
        if model_file is not None:
            self.load_weights(model_file)

        # initialization for test stage
        self._init_func = None
        self._step_func = None
        self._proj_func = None
        self._feat_shared = theano.shared(np.zeros((1, npatch, nimg)).astype(theano.config.floatX))
        self._scene_shared = theano.shared(np.zeros((1, ns)).astype(theano.config.floatX))

    def compute(self, state, w_idx, feat, scene):
        # word embedding
        word_vec = self.embedding.compute(w_idx)
        # split states
        e_tm1, c_tm1, h_tm1 = split_state(state, scheme=[(1, self.config['na']), (2, self.config['nh'])])
        # attention
        e_t, alpha = self.attention.compute(feat, T.concatenate([e_tm1, h_tm1, word_vec], axis=1))
        # lstm step
        e_w_s = T.concatenate([e_t, word_vec, scene], axis=-1)
        c_t, h_t = self.lstm.compute(e_w_s, c_tm1, h_tm1)
        # merge state
        new_state = T.concatenate([e_t, c_t, h_t], axis=-1)
        # add w_{t-1} as feature
        e_h_w_s = T.concatenate([e_t, h_t, word_vec, scene], axis=-1)
        # predict probability
        p = self.pred_mlp.compute(e_h_w_s)
        return new_state, p, alpha

    def scan_func(self, w_tm1, w_t, state, feat, scene):
        # update state
        new_state, p, alpha = self.compute(state, w_tm1, feat, scene)
        # cross-entropy loss
        loss = T.nnet.categorical_crossentropy(p, w_t)
        return new_state, p, loss, alpha

    def init_func(self, img_value, scene_value):
        if self._proj_func is None:
            img = T.tensor3()
            self._proj_func = theano.function([img], self.proj_mlp.compute(img))
        if self._init_func is None:
            init_e = self._feat_shared.mean(axis=1)
            init_state = T.concatenate([init_e, self.init_mlp.compute(init_e)], axis=-1)
            self._init_func = theano.function([], init_state)
        self._feat_shared.set_value(self._proj_func(img_value))
        self._scene_shared.set_value(scene_value)
        return self._init_func()

    def step_func(self, state_value, w_value):
        if self._step_func is None:
            w = T.ivector()
            state = T.matrix()
            new_state, p, _ = self.compute(state, w, self._feat_shared, self._scene_shared)
            self._step_func = theano.function([state, w], [new_state, T.log(p)])
        return self._step_func(state_value, w_value)

    def save_to_dir(self, save_dir, idx):
        save_file = osp.join(save_dir, self.name+'.h5.'+str(idx))
        for l in self.layers:
            l.save_weights(save_file)
        with h5py.File(save_file) as f:
            for k, v in self.config.items():
                f.attrs[k] = v

    def load_weights(self, model_file):
        for l in self.layers:
            l.load_weights(model_file)


if __name__ == '__main__':
    pass





