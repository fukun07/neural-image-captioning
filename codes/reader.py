import sys
import h5py
import json
import theano
import random
import os.path as osp
import numpy as np
import threading
from time import sleep
from config import DATA_ROOT


class Reader(threading.Thread):
    """
    This class is designed to automatically feed mini-batches.
    The reader constantly monitors the state of the variable 'data_buffer'.
    When finding the 'data_buffer' is None, the reader will fill a mini-batch into it.
    This is done in the backend, i.e. the reader is in an independent thread.
    For users, they only need to call iterate_batch() to get a new mini-batch.
    """
    daemon = True

    def __init__(self, batch_size=1,
                 data_split='train',
                 vocab_freq='freq5',
                 topic_type='gt',
                 stage='train',
                 data_dir=osp.join(DATA_ROOT, 'mscoco'),
                 feature_file='features_30res.h5',
                 topic_file='lda_80_topics.h5',
                 caption_switch='on',
                 feature_switch='on',
                 topic_switch='on',
                 head=0, tail=sys.maxint):
        # initialization of super class
        threading.Thread.__init__(self)
        # settings
        self.batch_size = batch_size
        self.data_split = data_split
        self.vocab_freq = vocab_freq
        self.topic_type = topic_type
        self.stage = stage
        # path
        self.data_dir = data_dir
        self.caption_file = osp.join(data_dir, 'processed_'+vocab_freq+'.json')
        self.image_dir = osp.join(self.data_dir, data_split+'2014')
        self.feature_file = osp.join(data_dir, 'features', feature_file)
        self.topic_file = osp.join(data_dir, 'topics', topic_file)
        # switches
        self.caption_switch = caption_switch
        self.feature_switch = feature_switch
        self.topic_switch = topic_switch

        # load image ids
        json_obj = json.load(open(self.caption_file, 'r'))
        self.image_ids = json_obj[self.data_split+'_image_ids']
        tail = min(tail, len(self.image_ids))
        self.image_ids = self.image_ids[head:tail]
        self.head = head
        self.tail = tail

        # vocabulary
        self.vocab = json_obj['vocab']
        # self.vocab = json.load(open(osp.join(data_dir, 'processed_freq10.json'), 'r'))['vocab']

        # load captions anyway
        captions = json_obj[data_split+'_captions']
        self.captions = [np.array(i).astype('int32') for i in captions][5*head:5*tail]

        # load features if necessary
        if self.feature_switch == 'on':
            with h5py.File(self.feature_file, 'r') as f:
                self.features = np.asarray(f[data_split][head:tail, ...])

        # load topics if necessary
        if self.topic_switch == 'on':
            with h5py.File(self.topic_file, 'r') as f:
                self.topics = np.asarray(f[data_split+'_'+topic_type][head:tail, ...])

        # initialization
        self.running = True
        self.data_buffer = None
        self.lock = threading.Lock()
        self.i_batch = 0
        self.n_caption = len(self.captions)
        self.n_image = self.n_caption / 5
        self.n_batch = None
        if self.stage == 'train':
            self.len_buckets = {}
            for i in xrange(self.n_caption):
                l = self.captions[i].shape[0]
                if l not in self.len_buckets:
                    self.len_buckets[l] = []
                self.len_buckets[l].append(i)
            self._shuffle()
        else:
            self.i_img = 0
        self.start()

    def _shuffle(self):
        """
        shuffle the data, called when one pass is over or at the beginning of training
        """
        self.data_flow = []
        for v in self.len_buckets.values():
            random.shuffle(v)
            for i in xrange(0, len(v), self.batch_size):
                self.data_flow.append(v[i:i+self.batch_size])
        random.shuffle(self.data_flow)
        self.n_batch = len(self.data_flow)

    def run(self):
        """
        over write the 'run' method of threading.Thread
        """
        while self.running:
            if self.data_buffer is None:
                if self.stage == 'train':
                    # reload a mini-batch if in train stage
                    index = self.data_flow[self.i_batch]
                    data = []
                    if self.caption_switch == 'on':
                        cap = [self.captions[i] for i in index]
                        data.append(np.transpose(np.vstack(cap)))
                    if self.feature_switch == 'on':
                        feature = [self.features[i/5, ...][None, ...] for i in index]
                        data.append(np.concatenate(feature, axis=0).astype(theano.config.floatX))
                    if self.topic_switch == 'on':
                        topic = [self.topics[i/5, ...][None, ...] for i in index]
                        data.append(np.concatenate(topic, axis=0).astype(theano.config.floatX))
                    self.i_batch += 1
                    if self.i_batch >= self.n_batch:
                        self.i_batch = 0
                        self._shuffle()
                else:
                    # sequentially iterate on single image if not in train stage
                    data = [self.image_ids[self.i_img]]
                    if self.feature_switch == 'on':
                        data.append(self.features[self.i_img, ...][None, ...].astype(theano.config.floatX))
                    if self.topic_switch == 'on':
                        data.append(self.topics[self.i_img, ...][None, ...].astype(theano.config.floatX))
                    self.i_img += 1
                    if self.i_img >= self.n_image:
                        self.i_img = 0
                # send data to buffer
                with self.lock:
                    self.data_buffer = data
            sleep(0.01)

    def iterate_batch(self):
        while self.data_buffer is None:
            # print 'wating for buffer ready'
            sleep(0.01)
        d = self.data_buffer
        with self.lock:
            self.data_buffer = None
        return d

    def close(self):
        self.running = False
        self.join()







