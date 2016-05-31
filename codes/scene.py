import os
import os.path as osp
import numpy as np
import h5py
import theano
from model.scene_mlp import SceneMlp
from optimizer import adam_optimizer
from config import DATA_ROOT, SAVE_ROOT


def train(dataset='mscoco', ensemble=10, mb=500, max_epoch=80, lr=0.0001):
    print '... loading topic'
    data_dir = osp.join(DATA_ROOT, dataset)
    with h5py.File(osp.join(data_dir, 'topics', 'lda_topics.h5')) as f:
        train_y = np.array(f['train_gt'], dtype=theano.config.floatX)
        valid_y = np.array(f['val_gt'], dtype=theano.config.floatX)[0:5000, :]
    print '... loading features'
    with h5py.File(osp.join(data_dir, 'features', 'features_1res.h5')) as f:
        train_x = np.array(f['train']).astype(theano.config.floatX)
        valid_x = np.array(f['val'][:5000, :]).astype(theano.config.floatX)

    N = train_x.shape[0]
    layer_sizes = [train_x.shape[-1], 512, 512, train_y.shape[-1]]
    save_dir = osp.join(SAVE_ROOT, dataset+'_scene_mlp')
    if osp.exists(save_dir):
        os.system('rm -rf {}'.format(save_dir))
    os.system('mkdir {}'.format(save_dir))
    for imodel in xrange(ensemble):
        print '... building model {}'.format(imodel)
        model = SceneMlp(layer_sizes=layer_sizes, name='scene_mlp')
        train_func, valid_func = adam_optimizer(model, lr=lr)
        best = 9999
        for epoch in xrange(max_epoch):
            index = np.random.permutation(N)
            for i in xrange(0, N, mb):
                train_func(train_x[index[i:i+mb]], train_y[index[i:i+mb]])
            loss, acc = valid_func(valid_x, valid_y)
            print 'model {}, epoch={}, loss: {}, acc: {}%'.format(imodel, epoch, loss, 100.*acc)
            if loss < best:
                model.save_to_dir(save_dir, imodel)
                best = loss
                print '** new best = {}'.format(best)


def predict(dataset='mscoco'):
    print '... initializing'
    data_dir = osp.join(DATA_ROOT, dataset)
    save_dir = osp.join(SAVE_ROOT, dataset+'_scene_mlp')

    model_files = [osp.join(save_dir, m) for m in os.listdir(save_dir) if 'h5' in m]
    funcs = []
    for model_file in model_files:
        model = SceneMlp(model_file=model_file, name='scene_mlp')
        funcs.append(theano.function(model.inputs[:1], model.log_proba))
    print '{}-ensemble'.format(len(funcs))

    print '... loading topic'
    data_y = {}
    with h5py.File(osp.join(data_dir, 'topics', 'lda_topics.h5')) as f:
        data_y['train'] = np.array(f['train_gt'], dtype=theano.config.floatX)
        data_y['val'] = np.array(f['val_gt'], dtype=theano.config.floatX)
        data_y['test'] = np.array(f['test_gt'], dtype=theano.config.floatX)
    print '... loading features'
    data_x = {}
    with h5py.File(osp.join(data_dir, 'features', 'features_1res.h5')) as f:
        data_x['train'] = np.array(f['train']).astype(theano.config.floatX)
        data_x['val'] = np.array(f['val']).astype(theano.config.floatX)
        data_x['test'] = np.array(f['test']).astype(theano.config.floatX)

    save_file = osp.join(data_dir, 'topics', 'lda_topics.h5')

    # predict topics
    for s in ['train', 'val', 'test']:
        print '... predicting {} set'.format(s)
        N = data_x[s].shape[0]
        for i in xrange(0, N, 1000):
            tail = min(N, i+1000)
            preds = [func(data_x[s][i:tail, :])[np.newaxis, :, :] for func in funcs]
            pred = np.exp(np.mean(np.concatenate(preds, axis=0), axis=0))
            pred = pred / pred.sum(axis=1)[:, np.newaxis]
            with h5py.File(save_file) as f:
                if '{}_pred2'.format(s) not in f:
                    f.create_dataset('{}_pred2'.format(s), shape=data_y[s].shape, dtype='float32')
                f['{}_pred2'.format(s)][i:tail, :] = pred


if __name__ == '__main__':
    train('flickr8k', 20)
    predict(dataset='flickr8k')
