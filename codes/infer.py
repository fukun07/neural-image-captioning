import os
import os.path as osp
import json
import numpy as np

from tools import Logger
from reader import Reader
from beam_search import BeamSearch
from tools import evaluate
from config import DATA_ROOT, SAVE_ROOT


def validate(beam_searcher, dataset, logger=None, res_file=None):
    if logger is None:
        logger = Logger(None)
    # generating captions
    all_candidates = []
    for i in xrange(dataset.n_image):
        data = dataset.iterate_batch()  # data: id, img, scene...
        sent = beam_searcher.generate(data[1:])
        cap = ' '.join([dataset.vocab[word] for word in sent])
        print '[{}], id={}, \t\t {}'.format(i, data[0], cap)
        all_candidates.append({'image_id': data[0], 'caption': cap})

    if res_file is None:
        res_file = 'tmp.json'
    json.dump(all_candidates, open(res_file, 'w'))
    gt_file = osp.join(dataset.data_dir, 'captions_'+dataset.data_split+'.json')
    scores = evaluate(gt_file, res_file, logger)
    if res_file == 'tmp.json':
        os.system('rm -rf %s' % res_file)

    return scores


if __name__ == '__main__':
    task = 'ss'
    print 'inference task: [{}]'.format(task)

    dataset = 'flickr8k'
    data_dir = osp.join(DATA_ROOT, dataset)

    if task == 'gnic':
        from model.gnic import Model
        model_list = ['mscoco-gnic-nh512-nw512-mb64-V8843/gnic.h5.merge']
        models = [Model(model_file=osp.join(SAVE_ROOT, m)) for m in model_list]

        valid_set = Reader(batch_size=1, data_split='test', vocab_freq='freq5', stage='test',
                           data_dir=data_dir, feature_file='features_1res.h5',
                           caption_switch='off', topic_switch='off', head=0, tail=1000)

        bs = BeamSearch(models, beam_size=3, num_cadidates=500, max_length=20)
        scores = validate(bs, valid_set)

    if task == 'ss':
        from model.ss import Model
        model_list = ['mscoco-nh512-nw512-mb64-V8843/ss.h5.merge']
        models = [Model(model_file=osp.join(SAVE_ROOT, m)) for m in model_list]

        valid_set = Reader(batch_size=1, data_split='test', vocab_freq='freq5', stage='val',
                           data_dir=data_dir, feature_file='features_1res.h5', topic_type='pred',
                           topic_file='lda_topics.h5', caption_switch='off', head=0, tail=1000)

        bs = BeamSearch(models, beam_size=3, num_cadidates=500, max_length=20)
        scores = validate(bs, valid_set)

    if task == 'ra':
        from model.ra import Model
        model_list = ['mscoco-ra-nh512-nw512-na512-mb64-V8843/ra.h5.merge']
        models = [Model(model_file=osp.join(SAVE_ROOT, m)) for m in model_list]

        valid_set = Reader(batch_size=1, data_split='test', vocab_freq='freq5', stage='test',
                           data_dir=data_dir, feature_file='features_30res.h5',
                           caption_switch='off', topic_switch='off', head=0, tail=1000)

        bs = BeamSearch(models, beam_size=3, num_cadidates=500, max_length=20)
        scores = validate(bs, valid_set)

    if task == 'rass':
        from model.rass import Model
        model_list = ['mscoco-rass-nh512-nw512-na512-mb64-V8843/rass.h5.merge']
        models = [Model(model_file=osp.join(SAVE_ROOT, m)) for m in model_list]

        valid_set = Reader(batch_size=1, data_split='test', vocab_freq='freq5', stage='val',
                           data_dir=data_dir, feature_file='features_30res.h5', topic_type='pred',
                           topic_file='lda_topics.h5', caption_switch='off', head=0, tail=1000)

        bs = BeamSearch(models, beam_size=3, num_cadidates=500, max_length=20)
        scores = validate(bs, valid_set)

    if task == 'evaluate':
        gt_file = osp.join(data_dir, 'captions_test.json')
        res_file = 'tmp.json'
        scores = evaluate(gt_file, res_file, None)
        for s in scores:
            print '{:.1f}'.format(100.*s)


