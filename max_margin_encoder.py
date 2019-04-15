"""Max-margin trained character-level encoder model.

Builds a model with two character-level LSTMs using the Dynet toolkit. Trained on parallel strings to maximize the cosine similarity between their representations, using a max-margin loss. The training data can be strings (graphemes or phonemes) or articulatory features from PanPhon.

Author: Shruti Rijhwani (srijhwan@cs.cmu.edu)
Last update: 2019-04-15
"""

from collections import defaultdict
import dynet as dy
import codecs
import random
import glob
import numpy as np
import sys
import argparse
import os.path
import time
import logging
import panphon as pp
from utils.constants import PANPHON_VECTOR_SIZE
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class MaxMarginEncoder():
    def __init__(self, embed_size, hidden_size, panphon, model_name, source_vocab_size, target_vocab_size, load_model=False):
        self.model_name = model_name
        self.model = dy.ParameterCollection()
        self.panphon = panphon

        if self.panphon:
            self.ws_panphon = self.model.add_parameters((embed_size, PANPHON_VECTOR_SIZE))
            self.bs_panphon = self.model.add_parameters((embed_size))

        self.source_lookup = self.model.add_lookup_parameters((source_vocab_size, embed_size))
        self.target_lookup = self.model.add_lookup_parameters((target_vocab_size, embed_size))

        self.source_lstm_forward = dy.LSTMBuilder(1, embed_size, hidden_size / 2, self.model)
        self.source_lstm_backward = dy.LSTMBuilder(1, embed_size, hidden_size / 2, self.model)
        self.target_lstm_forward = dy.LSTMBuilder(1, embed_size, hidden_size / 2, self.model)
        self.target_lstm_backward = dy.LSTMBuilder(1, embed_size, hidden_size / 2, self.model)

        # load model only if flag is true. will overwrite existing model if flag is false. set flag to True for fine-tuning or encoding/testing
        if load_model:
            try:
                self.model.populate(self.model_name)
                logging.info("Populated! " + self.model_name)
            except:
                sys.stderr.write("Model file %s not found!" % self.model_name)
                raise

    def save_model(self):
        self.model.save(self.model_name)

    def get_normalized_reps(self, embs, forward_lstm, backward_lstm, encode=False):
        word_reps = [dy.concatenate([forward_lstm.initial_state().transduce(emb)[-1],
                                     backward_lstm.initial_state().transduce(reversed(emb))[-1]]) for emb in embs]
        if not encode:
            return [dy.cdiv(rep, dy.l2_norm(rep)) for rep in word_reps]
        else:
            return [dy.cdiv(rep, dy.l2_norm(rep)).value() for rep in word_reps]

    def get_embedding(self, char, char_lookup):
        # parameter char will be panphon features if panphon (list of 22 integers) and will be character index if not.
        # parameter char_lookup not needed for panphon since transformation matrix is the same for source and target
        if self.panphon:
            w_panphon = dy.parameter(self.ws_panphon)
            b_panphon = dy.parameter(self.bs_panphon)
            return w_panphon * dy.inputVector(char) + b_panphon
        else:
            return char_lookup[char]

    def encode(self, entries, char_lookup, fwd, bwd):
        dy.renew_cg()
        embs = [[self.get_embedding(y, char_lookup) for y in temp] for temp in entries]
        return self.get_normalized_reps(embs, fwd, bwd, encode=True)

    def encode_source(self, entries):
        return self.encode(entries, self.source_lookup, self.source_lstm_forward, self.source_lstm_backward)

    def encode_target(self, entries):
        return self.encode(entries, self.target_lookup, self.target_lstm_forward, self.target_lstm_backward)

    def calculate_loss(self, words):
        dy.renew_cg()
        source_embs = [[self.get_embedding(x, self.source_lookup) for x in s] for s, t in words]
        target_embs = [[self.get_embedding(y, self.target_lookup) for y in t] for s, t in words]

        source_reps_norm = self.get_normalized_reps(source_embs, self.source_lstm_forward, self.source_lstm_backward)
        target_reps_norm = self.get_normalized_reps(target_embs, self.target_lstm_forward, self.target_lstm_backward)

        mtx_src = dy.concatenate_cols(source_reps_norm)
        mtx_trg = dy.concatenate_cols(target_reps_norm)
        similarity_mtx = dy.transpose(mtx_src) * mtx_trg
        loss = dy.hinge_dim(similarity_mtx, list(range(len(words))), d=1)

        return dy.sum_elems(loss) / (len(words) * len(words))
