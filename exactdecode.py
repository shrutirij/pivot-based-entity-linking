"""Exact match entity linking.

Takes an input entity and returns the knowledge base candidate that is an exact string match to the input. Optionally includes exact match to an intermediate pivot language, using Wikipedia language links to obtain the English KB candidate.

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
from utils.constants import DELIM,MAX_SCORE,ID_IDX,SOURCE_IDX,TARGET_IDX

class ExactDataLoader():
    def __init__(self, kb_filename, links_filename=None):
        self.kb = self.load_kb(kb_filename)
        if links_filename:
            self.links = self.load_links(links_filename)
        else:
            self.links = None

    def load_kb(self, filename):
        entries = {}
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(DELIM)
                if len(spl) != 2:
                    continue
                entries[spl[ID_IDX]] = int(spl[SOURCE_IDX])
        return entries
    
    def load_links(self,filename):
        entries = {}
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(DELIM)
                entries[spl[TARGET_IDX]] = int(spl[ID_IDX])
        return entries

class ExactDecode():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def decode(self, input_string, pivot=True):
        if input_string in self.data_loader.kb:
            return (self.data_loader.kb[input_string], MAX_SCORE)
        if pivot:
            assert self.data_loader.links != None
            if input_string in self.data_loader.links:
                return (self.data_loader.links[input_string], MAX_SCORE)
        return False

