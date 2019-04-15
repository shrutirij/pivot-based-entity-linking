"""Dataloader for a character-level entity similarity model. 

Builds character vocabulary from training data; converts strings to character indices (for embeddings) or to PanPhon feature vectors.

Author: Shruti Rijhwani (srijhwan@cs.cmu.edu)
Last update: 2019-04-15
"""

from collections import defaultdict
import codecs
import panphon as pp
from utils.constants import UNK,SOURCE_IDX,TARGET_IDX,PANPHON_VECTOR_SIZE,DELIM

class TrainDataLoader(object):
    def __init__(self, filename, panphon=False):
        self.panphon=panphon
        if self.panphon:
            self.ft = pp.FeatureTable()
            char_function = None
        else:
            self.source_vocab = defaultdict(lambda: len(self.source_vocab))
            self.target_vocab = defaultdict(lambda: len(self.target_vocab))
            char_function = self.char2int
            self.source_vocab[UNK]
            self.target_vocab[UNK]

        self.data = self.read_data(filename, char_function)
            
    def read_data(self, filename, char_function):
        parallel_data = []
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(DELIM)
                if self.panphon:
                    source = self.get_feature(spl[SOURCE_IDX])
                    target = self.get_feature(spl[TARGET_IDX])
                else:
                    source = [char_function(self.source_vocab, char) for char in spl[SOURCE_IDX]]
                    target = [char_function(self.target_vocab, char) for char in spl[TARGET_IDX]]
                if len(source) == 0 or len(target) == 0:
                    continue
                parallel_data.append((source, target))
        return parallel_data

    def char2int(self, vocab, char):
        """Builds vocabulary from training data"""
        return vocab[char]
    
    def char2int_unk(self, vocab, char):
        """Returns UNK if character not present in training data"""
        if char in vocab:
            return vocab[char]
        else:
            return vocab[UNK]
    
    def get_feature(self, word):
        """Returns articulatory feature vector using PanPhon"""
        pp_feats = self.ft.word_to_vector_list(word, numeric=True)

        # check if panphon has returned features, otherwise return default features (zeros)
        if len(pp_feats) > 0:
            return pp_feats
        return [[0] * PANPHON_VECTOR_SIZE] * len(word)

    def convert_source(self, entry):
        if self.panphon:
            return self.get_feature(entry)
        else:
            return [self.char2int_unk(self.source_vocab, char) for char in entry]

    def convert_target(self, entry):
        if self.panphon:
            return self.get_feature(entry)
        else:
            return [self.char2int_unk(self.target_vocab, char) for char in entry] 

    def convert_test_data(self, filename):
        return self.read_data(filename, self.char2int_unk)

    def source_vocab_size(self):
        if self.panphon:
            return 0
        return len(self.source_vocab)
    
    def target_vocab_size(self):
        if self.panphon:
            return 0
        return len(self.target_vocab)
