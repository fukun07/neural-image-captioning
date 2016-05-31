import json
import numpy as np
import copy
import heapq
import random
import os.path as osp

BEGIN_TOKEN = 0
END_TOKEN = 1


class Node(object):
    """
    node, used in BeamSearch
    """
    def __init__(self, states):
        self.states = states
        self.log_p = 0
        self.words = [BEGIN_TOKEN]


class BeamSearch(object):
    """
    This class implements the technique of beam search, which maintains several 
    best-by-now branches when generating captions.
    """
    def __init__(self, models, beam_size=1, num_cadidates=100, max_length=30):
        self.models = models
        self.beam_size = beam_size
        self.num_cadidates = num_cadidates
        self.max_length = max_length

    def generate(self, args):
        states = [m.init_func(*args) for m in self.models]
        best_by_now = [Node(states)]
        candidates = []
        length = 0

        while True:
            length += 1
            new_nodes = []
            for n in best_by_now:
                new_nodes.extend(self._get_new_nodes(n))
            best_by_now = sorted(new_nodes, key=lambda x: x.log_p)[-self.beam_size:]
            candidates.extend([n for n in best_by_now if n.words[-1] == END_TOKEN])
            best_by_now = [n for n in best_by_now if n.words[-1] != END_TOKEN]
            if len(best_by_now) < 1:
                break
            if len(candidates) >= self.num_cadidates:
                break
            if length > self.max_length:
                if len(candidates) < 1:
                    candidates = best_by_now
                break

        # scores = [c.log_p / (len(c.words)-1) for c in candidates]
        scores = [c.log_p for c in candidates]
        scores = np.array(scores)

        # return the best sentence, the #BEGIN# and #END# tokens are droped
        return candidates[scores.argmax()].words[1:-1]

    def _get_new_nodes(self, n):
        # compute hidden state and log-probability
        w = np.array([n.words[-1]]).astype('int32')
        states = []
        log_p = []
        for m, s in zip(self.models, n.states):
            new_s, p = m.step_func(s, w)
            states.append(new_s)
            log_p.append(p)
        log_p = np.vstack(log_p).mean(axis=0)

        # create beam_size branches
        new_nodes = []
        for i in xrange(self.beam_size):
            n2 = copy.deepcopy(n)
            n2.log_p += log_p.max()
            n2.words.append(log_p.argmax())
            n2.states = states
            log_p[log_p.argmax()] = -9999
            new_nodes.append(n2)
        return new_nodes


