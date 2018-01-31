import h5py
import theano
import theano.tensor as T
import os.path as osp
import numpy as np

from layers import split_state, MLP, BasicLSTM, Embedding, Attention


class Model(object):
    """
    Region Attention model
    """
    def __init__(self, name='ra', nimg=2048, na=512, nh=512, nw=512, nout=8843, npatch=30, model_file=None):
        self.name = name
        if model_file is not None:
            with h5py.File(model_file, 'r') as f:
                nimg = f.attrs['nimg']
                na = f.attrs['na']
                nh = f.attrs['nh']
                nw = f.attrs['nw']
                nout = f.attrs['nout']
                # npatch = f.attrs['npatch']
        self.config = {'nimg': nimg, 'na': na, 'nh': nh, 'nw': nw, 'nout': nout, 'npatch': npatch}

        # word embedding layer
        self.embedding = Embedding(n_emb=nout, dim_emb=nw, name=self.name+'@embedding')

        # initialization mlp layer
        self.init_mlp = MLP(layer_sizes=[na, 2*nh], output_type='tanh', name=self.name+'@init_mlp')
        self.proj_mlp = MLP(layer_sizes=[nimg, na], output_type='tanh', name=self.name+'@proj_mlp')

        # lstm
        self.lstm = BasicLSTM(dim_x=na+nw, dim_h=nh, name=self.name+'@lstm')

        # prediction mlp
        self.pred_mlp = MLP(layer_sizes=[na+nh+nw, nout], output_type='softmax', name=self.name+'@pred_mlp')

        # attention layer
        self.attention = Attention(dim_item=na, dim_context=na+nw+nh, hsize=nh, name=self.name+'@attention')

        # inputs
        cap = T.imatrix('cap')
        img = T.tensor3('img')
        self.inputs = [cap, img]

        # go through sequence
        feat = self.proj_mlp.compute(img)
        init_e = feat.mean(axis=1)
        init_state = T.concatenate([init_e, self.init_mlp.compute(init_e)], axis=-1)
        (state, self.p, loss, self.alpha), _ = theano.scan(fn=self.scan_func,
                                                           sequences=[cap[0:-1, :], cap[1:, :]],
                                                           outputs_info=[init_state, None, None, None],
                                                           non_sequences=[feat])

        # loss function
        loss = T.mean(loss)
        self.costs = [loss]

        # layers and parameters
        self.layers = [self.embedding, self.init_mlp, self.proj_mlp, self.attention, self.lstm, self.pred_mlp]
        self.params = sum([l.params for l in self.layers], [])

        # load weights from file, if model_file is not None
        if model_file is not None:
            self.load_weights(model_file)

        # these functions and variables are used in test stage
        self._init_func = None
        self._step_func = None
        self._proj_func = None
        self._feat_shared = theano.shared(np.zeros((1, npatch, na)).astype(theano.config.floatX))

    def compute(self, state, w_idx, feat):
        # word embedding
        word_vec = self.embedding.compute(w_idx)
        # split states
        e_tm1, c_tm1, h_tm1 = split_state(state, scheme=[(1, self.config['na']), (2, self.config['nh'])])
        # attention
        e_t, alpha = self.attention.compute(feat, T.concatenate([e_tm1, h_tm1, word_vec], axis=1))
        # lstm step
        e_w = T.concatenate([e_t, word_vec], axis=-1)
        c_t, h_t = self.lstm.compute(e_w, c_tm1, h_tm1)  # (mb,nh)
        # merge state
        new_state = T.concatenate([e_t, c_t, h_t], axis=-1)
        # predict word probability
        p = self.pred_mlp.compute(T.concatenate([e_t, h_t, word_vec], axis=-1))
        return new_state, p, alpha

    def scan_func(self, w_tm1, w_t, state, feat):
        # update state
        new_state, p, alpha = self.compute(state, w_tm1, feat)
        # cross-entropy loss
        loss = T.nnet.categorical_crossentropy(p, w_t)
        return new_state, p, loss, alpha

    def init_func(self, img_value):
        if self._proj_func is None:
            img = T.tensor3()
            self._proj_func = theano.function([img], self.proj_mlp.compute(img))
        if self._init_func is None:
            init_e = self._feat_shared.mean(axis=1)
            init_state = T.concatenate([init_e, self.init_mlp.compute(init_e)], axis=-1)
            self._init_func = theano.function([], init_state)
        self._feat_shared.set_value(self._proj_func(img_value))
        return self._init_func()

    def step_func(self, state_value, w_value):
        if self._step_func is None:
            w = T.ivector()
            state = T.matrix()
            new_state, p, _ = self.compute(state, w, self._feat_shared)
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



