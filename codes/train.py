import os
import os.path as osp
import json
import numpy as np

from reader import Reader
from beam_search import BeamSearch
from optimizer import adam_optimizer
from tools import evaluate, Timer, Logger, average_models
from config import DATA_ROOT, SAVE_ROOT


def train(model, beam_searcher, train_set, valid_set, save_dir, lr,
          display=100, starting=0, endding=20, validation=2000, patience=10, logger=None):
    """
    display:    output training infomation every 'display' mini-batches
    starting:   the starting snapshots, > 0 when resuming training
    endding:    the least training snapshots
    validation: evaluate on validation set every 'validation' mini-batches
    patience:       increase of endding when finds better model
    """
    train_func, _ = adam_optimizer(model, lr=lr)
    print '... training'
    logger = Logger(save_dir) if logger is None else logger
    timer = Timer()
    loss = 0
    imb = starting * validation
    best = -1
    best_snapshot = -1
    timer.tic()
    while imb < endding*validation:
        imb += 1
        x = train_set.iterate_batch()
        loss += train_func(*x)[0] / display
        if imb % display == 0:
            logger.info('snapshot={}, iter={},  loss={:.6f},  time={:.1f} sec'.format(imb/validation, imb, loss, timer.toc()))
            timer.tic()
            loss = 0
        if imb % validation == 0:
            saving_index = imb/validation
            model.save_to_dir(save_dir, saving_index)
            try:
                scores = validate(beam_searcher, valid_set, logger)
                if scores[3] > best:
                    best = scores[3]
                    best_snapshot = saving_index
                    endding = max(saving_index+patience, endding)
                logger.info('    ---- this Bleu-4 = [%.3f],   best Bleu-4 = [%.3f], endding -> %d' % \
                            (scores[3], best, endding))
            except OSError:
                print '[Ops!! OS Error]'

    logger.info('Training done, best snapshot is [%d]' % best_snapshot)
    return best_snapshot


def validate(beam_searcher, dataset, logger=None):
    # generating captions
    all_candidates = []
    for i in xrange(dataset.n_image):
        data = dataset.iterate_batch()  # data: id, img, scene...
        sent = beam_searcher.generate(data[1:])
        cap = u' '.join([dataset.vocab[word] for word in sent])
        # print data[0], cap
        all_candidates.append({u'image_id': data[0], 'caption': cap})

    res_file = '{:d}_tmp.json'.format(np.random.randint(0, 10000, 1)[0])
    json.dump(all_candidates, open(res_file, 'w'))
    gt_file = osp.join(dataset.data_dir, 'captions_'+dataset.data_split+'.json')
    scores = evaluate(gt_file, res_file, logger)
    os.system('rm -rf %s' % res_file)

    return scores


def find_last_snapshot(save_dir, resume_training=False):
    """
    This function finds the latest model in save_dir.
    If resume_training is set to True, it will return the file name of the latest model,
    otherwise it will delete the existing save_dir and create a clean one.
    """
    if not resume_training:
        print '... resume option: Create New Model'
        if os.path.exists(save_dir):
            os.system('rm -rf {}'.format(save_dir))
        os.mkdir(save_dir)
        return None, 0
    if not os.path.exists(save_dir):
        print '... resume option: Previous Model Not Found'
        os.mkdir(save_dir)
        return None, 0
    files = os.listdir(save_dir)
    m = 0
    model_file = None
    for f in files:
        s = f.split('.')
        if not s[-2] == 'h5':
            continue
        if s[-1] != 'merge' and m < int(s[-1]):
            m = int(s[-1])
            model_file = f
    print '... resume option: Continue with Previous Model [{}]'.format(model_file)
    if model_file is None:
        return None, 0
    return os.path.join(save_dir, model_file), m


if __name__ == '__main__':
    # choose a task to train
    task = 'ss'
    print 'train task [{}]'.format(task)

    # choose a dataset to train (mscoco, flickr8k, flickr30k)
    dataset = 'mscoco'
    data_dir = osp.join(DATA_ROOT, dataset)

    if task == 'gnic':
        from model.gnic import Model
        # settings
        mb = 64  # mini-batch size
        lr = 0.0001  # learning rate
        nh = 512  # size of LSTM's hidden size
        nw = 512  # size of word embedding vector
        name = 'gnic'  # model name, just setting it to 'gnic' is ok. 'gnic'='google nic'
        vocab_freq = 'freq5'  # use the vocabulary that filtered out words whose frequences are less than 5

        print '... loading data {}'.format(dataset)
        train_set = Reader(batch_size=mb, data_split='train', vocab_freq=vocab_freq, stage='train',
                           data_dir=data_dir, feature_file='features_1res.h5',
                           topic_switch='off')
        valid_set = Reader(batch_size=1, data_split='val', vocab_freq=vocab_freq, stage='val',
                           data_dir=data_dir, feature_file='features_1res.h5',
                           caption_switch='off', topic_switch='off')

        nimg = train_set.features.shape[-1]
        nout = len(train_set.vocab)
        save_dir = '{}-{}-nh{}-nw{}-mb{}-V{}'.\
            format(dataset.lower(), task, nh, nw, mb, nout)
        save_dir = osp.join(SAVE_ROOT, save_dir)

        model_file, m = find_last_snapshot(save_dir, resume_training=True)
        os.system('cp model/gnic.py {}/'.format(save_dir))  # backup the model description
        logger = Logger(save_dir)
        logger.info('... building')
        model = Model(name=name, nimg=nimg, nh=nh, nw=nw, nout=nout, model_file=model_file)

    if task == 'ss':
        from model.ss import Model
        # settings
        mb = 64  # mini-batch size
        lr = 0.0001  # learning rate
        nh = 512  # size of LSTM's hidden size
        nw = 512  # size of word embedding vector
        name = 'ss'  # model name, just setting it to 'sf' is ok. 'sf'='scene factor'
        vocab_freq = 'freq5'  # use the vocabulary that filtered out words whose frequences are less than 5

        print '... loading data {}'.format(dataset)
        train_set = Reader(batch_size=mb, data_split='train', vocab_freq=vocab_freq, stage='train',
                           data_dir=data_dir, feature_file='features_1res.h5', topic_file='lda_topics.h5')
        valid_set = Reader(batch_size=1, data_split='val', vocab_freq=vocab_freq, stage='val',
                           data_dir=data_dir, feature_file='features_1res.h5', topic_file='lda_topics.h5',
                           topic_type='pred', caption_switch='off')

        nimg = train_set.features.shape[-1]
        ns = train_set.topics.shape[-1]
        nout = len(train_set.vocab)
        save_dir = '{}-{}-nh{}-nw{}-mb{}-V{}'.\
            format(dataset.lower(), task, nh, nw, mb, nout)
        save_dir = osp.join(SAVE_ROOT, save_dir)

        model_file, m = find_last_snapshot(save_dir, resume_training=True)
        os.system('cp model/ss.py {}/'.format(save_dir))
        logger = Logger(save_dir)
        logger.info('... building')
        model = Model(name=name, nimg=nimg, nh=nh, nw=nw, nout=nout, ns=ns, model_file=model_file)

    if task == 'ra':
        from model.ra import Model
        # settings
        mb = 64  # mini-batch size
        lr = 0.0001  # learning rate
        nh = 512  # size of LSTM's hidden size
        nw = 512  # size of word embedding vector
        na = 512  # size of the region features after dimensionality reduction
        name = 'ra'  # model name, just setting it to 'ra' is ok. 'ra'='region attention'
        vocab_freq = 'freq5'  # use the vocabulary that filtered out words whose frequences are less than 5

        print '... loading data {}'.format(dataset)
        train_set = Reader(batch_size=mb, data_split='train', vocab_freq=vocab_freq, stage='train',
                           data_dir=data_dir, feature_file='features_30res.h5', topic_switch='off')
        valid_set = Reader(batch_size=1, data_split='val', vocab_freq=vocab_freq, stage='val',
                           data_dir=data_dir, feature_file='features_30res.h5',
                           caption_switch='off', topic_switch='off')

        npatch, nimg = train_set.features.shape[1:]
        nout = len(train_set.vocab)
        save_dir = '{}-{}-nh{}-nw{}-na{}-mb{}-V{}'.\
            format(dataset.lower(), task, nh, nw, na, mb, nout)
        save_dir = osp.join(SAVE_ROOT, save_dir)

        model_file, m = find_last_snapshot(save_dir, resume_training=True)
        os.system('cp model/ra.py {}/'.format(save_dir))
        logger = Logger(save_dir)
        logger.info('... building')
        model = Model(name=name, nimg=nimg, nh=nh, na=na, nw=nw, nout=nout, npatch=npatch, model_file=model_file)

    if task == 'rass':
        from model.rass import Model
        # settings
        mb = 64  # mini-batch size
        lr = 0.0001  # learning rate
        nh = 512  # size of LSTM's hidden size
        nw = 512  # size of word embedding vector
        na = 512  # size of the region features after dimensionality reduction
        name = 'rass'  # model name, just setting it to 'sfra' is ok. 'sfra'='sf + ra'
        vocab_freq = 'freq5'  # use the vocabulary that filtered out words whose frequences are less than 5

        print '... loading data {}'.format(dataset)
        train_set = Reader(batch_size=mb, data_split='train', vocab_freq=vocab_freq, stage='train',
                           data_dir=data_dir, feature_file='features_30res.h5', topic_file='lda_topics.h5',
                           head=0)
        valid_set = Reader(batch_size=1, data_split='val', vocab_freq=vocab_freq, stage='val',
                           data_dir=data_dir, feature_file='features_30res.h5', topic_file='lda_topics.h5',
                           topic_type='pred', caption_switch='off')

        nimg = train_set.features.shape[-1]
        ns = train_set.topics.shape[-1]
        nout = len(train_set.vocab)
        save_dir = '{}-{}-nh{}-nw{}-na{}-mb{}-V{}'.\
            format(dataset.lower(), task, nh, nw, na, mb, nout)
        save_dir = osp.join(SAVE_ROOT, save_dir)

        model_file, m = find_last_snapshot(save_dir, resume_training=True)
        os.system('cp model/rass.py {}/'.format(save_dir))
        logger = Logger(save_dir)
        logger.info('... building')
        model = Model(name=name, nimg=nimg, nh=nh, nw=nw, na=na, nout=nout, ns=ns, model_file=model_file)

    # start training
    bs = BeamSearch([model], beam_size=1, num_cadidates=100, max_length=20)
    best = train(model, bs, train_set, valid_set, save_dir, lr,
                 display=100, starting=m, endding=20, validation=2000, patience=10, logger=logger)
    average_models(best=best, L=6, model_dir=save_dir, model_name=name+'.h5')

