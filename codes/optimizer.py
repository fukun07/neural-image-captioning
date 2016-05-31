import theano
import theano.tensor as T
import numpy as np


def adam_optimizer(model, lr=0.0002, beta1=0.1, beta2=0.001, epsilon=1e-8, gamma=1-1e-8):
    updates = []
    lr = theano.shared(np.array(lr).astype(theano.config.floatX))

    # compute gradients
    grads = theano.grad(model.costs[0], model.params)

    # define updating rules
    i = theano.shared(np.float32(1))
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(model.params, grads):
        m = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        m_t = (beta1_t * g) + ((1. - beta1_t) * m)
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    train_func = theano.function(model.inputs, model.costs, updates=updates)
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    valid_func = theano.function(model.inputs, model.costs)

    return train_func, valid_func


def sgd_optimizer(model, lr=0.001, momentum=0.9):
    lr = theano.shared(np.array(lr).astype(theano.config.floatX))
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # the updates of SGD with momentum
    updates = []
    grads = T.grad(model.costs[0], model.params)
    for param, grad in zip(model.params, grads):
        param_update = theano.shared(param.get_value()*0.)
        updates.append((param, param - lr * param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*grad))

    train_func = theano.function(model.inputs, model.costs, updates=updates)
    valid_func = theano.function(model.inputs, model.costs)

    return train_func, valid_func



