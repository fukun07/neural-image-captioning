import h5py
import numpy as np
import theano
import theano.tensor as T

"""
This file includes some basic layers that are shared among different models
"""


def shared_uniform(shape, scale=0.01, name=None):
    value = np.random.uniform(-scale, scale, shape).astype(theano.config.floatX)
    var = theano.shared(value)
    if name is not None:
        var.name = name
    return var


def shared_randn(shape, scale=0.01, name=None):
    value = scale * np.random.randn(*shape).astype(theano.config.floatX)
    var = theano.shared(value)
    if name is not None:
        var.name = name
    return var


def split_state(state, scheme):
        ch = []
        offset = 0
        for num, len in scheme:
            for i in xrange(num):
                ch.append(state[:, offset:offset+len])
                offset += len
        return ch


class Layer(object):
    """
    The super class of different layers
    """
    def __init__(self, name):
        self.name = name
        self.params = []

    def load_weights(self, model_file):
        with h5py.File(model_file, 'r') as f:
            for p in self.params:
                if p.name in f:
                    p.set_value(np.array(f[p.name]).astype(theano.config.floatX))
                else:
                    print 'parameter [ {} ] not found in save file'.format(p.name)

    def save_weights(self, model_file):
        with h5py.File(model_file) as f:
            for p in self.params:
                if p.name in f:
                    del f[p.name]
                f[p.name] = p.get_value().astype(theano.config.floatX)


class MLP(Layer):
    """
    An inplementation of multilayer perceptron
    """
    def __init__(self, layer_sizes, activation_type='tanh', output_type='tanh', name='mlp'):
        Layer.__init__(self, name)
        self.layer_sizes = layer_sizes
        self.activation_type = activation_type
        self.output_type = output_type

        # initialize parameters
        self.w = []
        self.b = []
        for i in xrange(len(layer_sizes)-1):
            self.w.append(shared_uniform(layer_sizes[i:i+2], name=self.name+'#w'+str(i)))
            self.b.append(shared_uniform(layer_sizes[i+1], scale=0., name=self.name+'#b'+str(i)))

        # record params
        self.params = self.w + self.b

    def compute(self, x):
        self.h = [x]

        # choose activation function
        if self.activation_type == 'tanh':
            activation = T.tanh
        elif self.activation_type == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation_type == 'relu':
            activation = T.nnet.relu
        else:
            raise TypeError('Unsupported MLP activation Type !!')

        # forward
        for i in xrange(len(self.layer_sizes)-2):
            self.h.append(activation(T.dot(self.h[-1], self.w[i]) + self.b[i]))
        self.z = T.dot(self.h[-1], self.w[-1]) + self.b[-1]

        # choose the activation of last layer
        if self.output_type == 'tanh':
            self.h.append(T.tanh(self.z))
        elif self.output_type == 'softmax':
            self.h.append(T.nnet.softmax(self.z))
        elif self.output_type == 'sigmoid':
            self.h.append(T.nnet.sigmoid(self.z))
        elif self.output_type == 'relu':
            self.h.append(T.nnet.relu(self.z))
        elif self.output_type == 'linear':
            self.h.append(self.z)
        else:
            raise TypeError('Unsupported MLP output Type !!')

        return self.h[-1]


class BasicLSTM(Layer):
    """
    An inplement of basic LSTM
    """
    def __init__(self, dim_x, dim_h, name='basic_lstm'):
        Layer.__init__(self, name)

        # initialize parameters
        self.w = shared_randn([dim_x+dim_h, 4*dim_h], scale=0.01, name=self.name+'#w')
        self.b = shared_uniform(4*dim_h, scale=0., name=self.name+'#b')
        b_value = self.b.get_value()
        b_value[3*dim_h:] += 3. # initialize forget gate bias to a large number
        self.b.set_value(b_value)

        # record the size information
        self.dim_h = self.w.get_value().shape[1] / 4

        # add all above into params
        self.params = [self.w, self.b]

    def compute(self, x_t, c_tm1, h_tm1):
        x_and_h = T.concatenate([x_t, h_tm1], axis=1)  # x:(mb, dim_x),  h:(mb,dim_h)
        state = T.dot(x_and_h, self.w) + self.b  # split state to (c, i, o, f)
        c_tilde = T.tanh(state[:, 0:self.dim_h])
        i_t = T.nnet.sigmoid(state[:, self.dim_h:2*self.dim_h])
        o_t = T.nnet.sigmoid(state[:, 2*self.dim_h:3*self.dim_h])
        f_t = T.nnet.sigmoid(state[:, 3*self.dim_h:4*self.dim_h])
        c_t = i_t * c_tilde + f_t * c_tm1
        h_t = o_t * T.tanh(c_t)
        return c_t, h_t


class Attention(Layer):
    """
    An implement of attention layer
    """
    def __init__(self, dim_item, dim_context, hsize, name='attention'):
        Layer.__init__(self, name)

        # initialize parameters
        self.w_item = shared_uniform([dim_item, hsize], name=self.name+'#w_item')
        self.w_context = shared_uniform([dim_context, hsize], name=self.name+'#w_context')
        self.b = shared_uniform(hsize, scale=0., name=self.name+'#b')
        self.w_score = shared_uniform([hsize, 1], name=self.name+'#w_score')

        # add all above into params
        self.params = [self.w_item, self.w_context, self.b, self.w_score]

    def compute(self, item, context):
        item_h = T.dot(item, self.w_item)  # item:(mb, #item, dim_item)
        context_h = T.dot(context, self.w_context)  # context:(mb, dim_context)
        h = T.tanh(item_h + context_h[:, None, :] + self.b)
        score = T.dot(h, self.w_score).flatten(ndim=2)
        alpha = T.nnet.softmax(score)  # alpha:(mb, #item)
        att = (item * alpha[:, :, None]).sum(axis=1)
        return att, alpha


class Embedding(Layer):
    """
    An implement of embedding layer
    """
    def __init__(self, n_emb, dim_emb, name='embedding'):
        Layer.__init__(self, name)

        # initialize parameters
        self.table = shared_uniform([n_emb, dim_emb], name=self.name+'#table')

        # add all above into params
        self.params = [self.table]

    def compute(self, index):
        return self.table[index, :]
