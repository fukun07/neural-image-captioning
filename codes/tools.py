import time
import os
import os.path as osp
import numpy as np
import json
import h5py
import logging
from collections import Counter
from skimage.io import imread
from pycoco.bleu.bleu import Bleu
from pycoco.meteor.meteor import Meteor
from pycoco.rouge.rouge import Rouge
from pycoco.cider.cider import Cider
from config import DATA_ROOT, SAVE_ROOT


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.avg_time = 0
        self.n_toc = 0

    def tic(self):
        self.n_toc = 0
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.n_toc += 1.
        self.avg_time = self.total_time / self.n_toc
        return self.total_time


class Logger:
    """
    When receiving a message, first print it on screen, then write it into log file.
    If save_dir is None, it writes no log and only prints on screen.
    """
    def __init__(self, save_dir):
        if save_dir is not None:
            self.logger = logging.getLogger()
            logging.basicConfig(filename=osp.join(save_dir, 'experiment.log'), format='%(asctime)s |  %(message)s')
            logging.root.setLevel(level=logging.INFO)
        else:
            self.logger = None

    def info(self, msg, to_file=True):
        print msg
        if self.logger is not None and to_file:
            self.logger.info(msg)


def evaluate(gt_file, re_file, logger=None):
    """
    This function is reformed from MSCOCO evaluating code.
    The reference sentences are read from gt_file, 
    the generated sentences to be evaluated are read from res_file

    """
    gts = json.load(open(gt_file, 'r'))
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    metrics = []
    res = json.load(open(re_file, 'r'))
    res = {c['image_id']: [c['caption']] for c in res}
    gts = {k: v for k, v in zip(gts['image_ids'], gts['captions']) if k in res}
    for scorer, method in scorers:
        if logger is not None:
            logger.info('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                if logger is not None:
                    logger.info("%s: %0.3f" % (m, sc))
            metrics.extend(score)
        else:
            if logger is not None:
                logger.info("%s: %0.3f" % (method, score))
            metrics.append(score)
    return metrics


def average_models(best, L=6, model_dir='', model_name='ra.h5'):
    print '... merging'
    print '{} {:d}-{:d}'.format(model_dir, best-L/2, best+L/2)
    params = {}
    side_info = {}
    attrs = {}
    for i in xrange(best-L/2, best+L/2):
        with h5py.File(osp.join(model_dir, model_name+'.'+str(i)), 'r') as f:
            for k, v in f.attrs.items():
                attrs[k] = v
            for p in f.keys():
                if '#' not in p:
                    side_info[p] = f[p][...]
                elif p in params:
                    params[p] += np.array(f[p]).astype('float32') / L
                else:
                    params[p] = np.array(f[p]).astype('float32') / L
    with h5py.File(osp.join(model_dir, model_name+'.merge'), 'w') as f:
        for p in params.keys():
            f[p] = params[p]
        for s in side_info.keys():
            f[s] = side_info[s]
        for k, v in attrs.items():
            f.attrs[k] = v


def word_count(captions, list_depth=2):
    word_set = []
    if list_depth == 1:
        for cap in captions:
            word_set.extend(cap.split(' '))
    elif list_depth == 2:
        for caps in captions:
            for cap in caps:
                word_set.extend(cap.split(' '))
    else:
        raise ValueError('list depth should be 1 or 2')

    return Counter(word_set).items()


def read_coco_image(data_split, coco_id, root_dir=osp.join(DATA_ROOT, 'mscoco')):
    file_name = 'COCO_{}2014_'.format(data_split) + str(coco_id).zfill(12) + '.jpg'
    im = imread(osp.join(root_dir, data_split+'2014', file_name))
    return im


if __name__ == '__main__':
    average_models(best=36, L=10,
                   model_dir=osp.join(SAVE_ROOT, 'flickr8k-rass-nh512-nw128-na128-mb64-V2547-r5'),
                   model_name='rass.h5')
