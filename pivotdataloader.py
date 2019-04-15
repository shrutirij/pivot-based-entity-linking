"""Dataloader for pivot-based-entity-linking. 

Encodes the knowledge base and pivoting language links using a trained entity similarity model.

Author: Shruti Rijhwani (srijhwan@cs.cmu.edu)
Last update: 2019-04-15
"""

import codecs
from traindataloader import TrainDataLoader
from max_margin_encoder import MaxMarginEncoder
import numpy as np
import sys
import logging
from utils.constants import ID_IDX,SOURCE_IDX,TARGET_IDX,DEFAULT_ENCODE_BATCH_SIZE,DELIM
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class PivotDataLoader(object):
    def __init__(self, kb_filename, links_filename=None, kb_encoding_path="kb.encode", links_encoding_path="links.encode", training_data_loader=None, encoder_model=None, load_encodings=False):
        self.kb, kb_entries = self.load_kb(kb_filename)
        if links_filename:
            self.links, links_entries = self.load_links(links_filename)
        else:
            self.links = None

        if encoder_model and not load_encodings:
            self.kb_encodings = self.batch_encode(kb_entries, encode_func=encoder_model.encode_source, convert_func=training_data_loader.convert_source, encoding_path=kb_encoding_path)
            np.savez_compressed(kb_encoding_path, arr=self.kb_encodings)
            if self.links:
                self.links_encodings = self.batch_encode(links_entries, encode_func=encoder_model.encode_target, convert_func=training_data_loader.convert_target, encoding_path=links_encoding_path)
                np.savez_compressed(links_encoding_path, arr=self.links_encodings)
        else:
            try:
                self.kb_encodings = np.load(kb_encoding_path + '.npz')['arr']
            except IOError:
                sys.stderr.write("KB encodings not found!\n")

            if self.links:
                try:
                    self.links_encodings = np.load(links_encoding_path + '.npz')['arr']
                except IOError:
                    sys.stderr.write("Links encodings not found!\n")
    
    def load_kb(self, filename):
        db = []
        entries = []
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(DELIM)
                if len(spl) != 3:
                    continue
                db.append(int(spl[ID_IDX]))
                entries.append(spl[SOURCE_IDX])
        return db, entries

    def load_links(self, filename):
        links = []
        entries = []
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(DELIM)
                links.append(int(spl[ID_IDX]))
                entries.append(spl[TARGET_IDX])
        return links, entries

    def batch_encode(self, entries, encode_func, convert_func, encoding_path):
        encoded = []
        for i in range(0, len(entries), DEFAULT_ENCODE_BATCH_SIZE):
            logging.info("Read %s entries" %i)
            cur_size = min(DEFAULT_ENCODE_BATCH_SIZE, len(entries) - i)
            encoded += encode_func([convert_func(entry) for entry in entries[i:i+cur_size]])
        return np.array(encoded)
    