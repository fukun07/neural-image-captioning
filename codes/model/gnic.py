import h5py
import theano
import theano.tensor as T
import os.path as osp

from layers import split_state, MLP, BasicLSTM, Embedding


class Model(object):
    """
    an re-implementation of google NIC system, used as the baseline in our paper
    """
    def __init__(self, name='gnic', nimg=2048, nh=512, nw=512, nout=8843, model_file=None):
        self.name = name
        if model_file is not None:
            with h5py.File(model_file, 'r') as f:
                nimg = f.attrs['nimg']
                nh = f.attrs['nh']
                nw = f.attrs['nw']
                nout = f.attrs['nout']
        self.config = {'nimg': nimg, 'nh': nh, 'nw': nw, 'nout': nout}

        # word embedding layer
        self.embedding = Embedding(n_emb=nout, dim_emb=nw, name=self.name+'@embedding')

        # initialization mlp layer
        self.proj_mlp = MLP(layer_sizes=[nimg, 2*nh], output_type='tanh', name=self.name+'@proj_mlp')

        # lstm
        self.lstm = BasicLSTM(dim_x=nw, dim_h=nh, name=self.name+'@lstm')

        # prediction mlp
        self.pred_mlp = MLP(layer_sizes=[nh+nw, nout], output_type='softmax', name=self.name+'@pred_mlp')

        # inputs
        cap = T.imatrix('cap')
        img = T.matrix('img')
        self.inputs = [cap, img]

        # go through sequence
        init_state = self.proj_mlp.compute(img)
        (state, self.p, loss), _ = theano.scan(fn=self.scan_func,
                                               sequences=[cap[0:-1, :], cap[1:, :]],
                                               outputs_info=[init_state, None, None])

        # loss function
        loss = T.mean(loss)
        self.costs = [loss]

        # layers and parameters
        self.layers = [self.embedding, self.proj_mlp, self.lstm, self.pred_mlp]
        self.params = sum([l.params for l in self.layers], [])

        # load weights from file, if model_file is not None
        if model_file is not None:
            self.load_weights(model_file)

        # these functions are used in test stage
        self._init_func = None
        self._step_func = None

    def compute(self, state, w_idx):
        # word embedding
        word_vec = self.embedding.compute(w_idx)
        # split states
        c_tm1, h_tm1 = split_state(state, scheme=[(2, self.config['nh'])])
        # lstm step
        c_t, h_t = self.lstm.compute(word_vec, c_tm1, h_tm1)
        # merge state
        new_state = T.concatenate([c_t, h_t], axis=-1)
        # add w_{t-1} as feature
        h_and_w = T.concatenate([h_t, word_vec], axis=-1)
        # predict probability
        p = self.pred_mlp.compute(h_and_w)
        return new_state, p

    def scan_func(self, w_tm1, w_t, state):
        # update state
        new_state, p = self.compute(state, w_tm1)
        # cross-entropy loss
        loss = T.nnet.categorical_crossentropy(p, w_t)
        return new_state, p, loss

    def init_func(self, img_value):
        if self._init_func is None:
            img = T.matrix()
            init_state = self.proj_mlp.compute(img)
            self._init_func = theano.function([img], init_state)
        return self._init_func(img_value)

    def step_func(self, state_value, w_value):
        if self._step_func is None:
            w = T.ivector()
            state = T.matrix()
            new_state, p = self.compute(state, w)
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


