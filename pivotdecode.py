"""Pivot-based entity linking decoder.

Takes an input entity and returns the top-k candidate entries from the knowledge base, optionally with pivoting through a high-resource language for better performance -- see https://arxiv.org/abs/1811.04154 for details.

Author: Shruti Rijhwani (srijhwan@cs.cmu.edu)
Last update: 2019-04-15
"""

from collections import defaultdict
import codecs
import numpy as np
import sys
import argparse
import time
from collections import OrderedDict
from utils.constants import MAX_SCORE

class PivotDecode():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_ranks(self, cur_scores, topk):
        limit = min([len(cur_scores),topk*2])
        max_idx = np.argpartition(cur_scores, -limit)[-limit:]
        ranked_idxs = max_idx[np.argsort(cur_scores[max_idx])]
        return ranked_idxs

    def get_predictions(self, ranked_ids, ranked_scores, exact, topk):
        paired_scores = list(zip(ranked_ids, ranked_scores))
        if exact:
            paired_scores = [exact] + paired_scores
        sorted_scores = sorted(paired_scores, key = lambda x: x[1])
        sorted_dedup_scores = sorted(dict(sorted_scores).items(), key=lambda x: x[1], reverse=True)
        pred_len = min(topk, len(paired_scores))
        return sorted_dedup_scores[:pred_len]

    def decode(self, input_encoding, pivot=True, exact=False, topk=200):
        scores = input_encoding.dot(self.data_loader.kb_encodings.T)

        if pivot:
            entity_ids = np.concatenate((self.data_loader.kb, self.data_loader.links))
            link_scores = input_encoding.dot(self.data_loader.links_encodings.T)
            scores = np.concatenate((scores, link_scores))
        else:
            entity_ids = np.array(self.data_loader.kb)

        ranked_idxs = self.get_ranks(scores, topk)
        ranked_ids = entity_ids[ranked_idxs][::-1]
        ranked_scores = scores[ranked_idxs][::-1]

        return self.get_predictions(ranked_ids, ranked_scores, exact, topk)
