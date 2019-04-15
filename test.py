"""Entity linking script for pivot-based-entity-linking.

Takes an input file of strings, encodes them using the trained encoder and returns the top-k most similar candidates from the knowledge base.

Author: Shruti Rijhwani (srijhwan@cs.cmu.edu)
Last update: 2019-04-15
"""

from collections import defaultdict
import codecs
import numpy as np
import sys
import argparse
import time
import logging
from collections import OrderedDict
from pivotdataloader import PivotDataLoader
from max_margin_encoder import MaxMarginEncoder
from traindataloader import TrainDataLoader
from pivotdecode import PivotDecode
from exactdecode import ExactDataLoader, ExactDecode
import datetime
from utils.constants import NOPIVOT_OUT,PIVOT_OUT,DEFAULT_ENCODE_BATCH_SIZE,ID_IDX,INPUT_IDX,TEST_IDX,DELIM
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class PivotELTest():
    def __init__(self, input_file, encoder_model, train_data_loader, pivotdecoder, exactdecoder=None, testing=True):
        self.testing = testing
        if self.testing:
            self.test_ids, self.input_entries = self.read_test_data(input_file)
        else:
            self.input_entries = self.read_input_data(input_file) 
        
        # encodes test encodings. if saved using np.save(), these can be loaded using np.load()
        encoded = []
        for i in range(0, len(self.input_entries), DEFAULT_ENCODE_BATCH_SIZE):
            logging.info("Read %s entries" %i)
            cur_size = min(DEFAULT_ENCODE_BATCH_SIZE, len(self.input_entries) - i)
            encoded += encoder_model.encode_target([train_data_loader.convert_target(entry) for entry in self.input_entries[i:i+cur_size]])
        self.test_encodings = np.array(encoded)
        self.pivotdecoder = pivotdecoder
        self.exactdecoder = exactdecoder
               
    def read_test_data(self, filename):
        ids = []
        entries = []
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(DELIM)
                ids.append(int(spl[ID_IDX]))
                entries.append(spl[TEST_IDX])
        return ids, entries

    def read_input_data(self, filename):
        entries = []
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(DELIM)
                entries.append(spl[INPUT_IDX])
        return entries

    def update_recalls(self, recalls, predicted, true):
        if true == predicted[0]:
            recalls[1] += 1
        if true in predicted[:5]:
            recalls[5] += 1
        if true in predicted[:10]:
            recalls[10] += 1
        if true in predicted[:20]:
            recalls[20] += 1
        if true in predicted[:100]:
            recalls[100] += 1

    def write_recalls(self, recalls):
        for i in [1, 5, 10, 20, 100]:
            logging.info("Recall at %d: %0.4f" %(i,(recalls[i]/len(self.input_entries))))

    def test_pivot(self, pivot=True, topk=30):
        topk_preds = []
        if self.testing:
            recalls = defaultdict(lambda: 0.0)
        
        for idx, input_string in enumerate(self.input_entries):
            cur_encoding = self.test_encodings[idx]
            if self.exactdecoder:
                exact_match = self.exactdecoder.decode(input_string=input_string, pivot=pivot)
            else:
                exact_match = False
            predictions = self.pivotdecoder.decode(input_encoding=cur_encoding, pivot=pivot, exact=exact_match, topk=topk)
            topk_preds.append(predictions)
            if self.testing:
                self.update_recalls(recalls, [kb_idx for (kb_idx, score) in predictions], self.test_ids[idx])
        if self.testing:
            self.write_recalls(recalls)

        return topk_preds

def write_topk(filepointer, preds):
    """Write out candidate KB IDs for each input entity separated by '***'"""
    for pred in preds:
        for kb_idx, score in pred:
            filepointer.write("%d ||| %0.4f\n" %(kb_idx,score))
        filepointer.write("***\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--embed_size', type=int, default=64, help='Saved model character embedding size')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Saved model character LSTM hidden size')
    parser.add_argument('--train_file', help='Location of training file used for the model dataloader')
    parser.add_argument('--model_file', help='Location of saved model')
    parser.add_argument('--panphon', action='store_true', default=False, help='Use if PanPhon was used for training the model')

    parser.add_argument('--kb', help='Location of knowledge base file with ID and English title')
    parser.add_argument('--links', help='Location of English-HRL links file for pivoting')
    parser.add_argument('--kb_encodings', help="Location to store encoded KB")
    parser.add_argument('--links_encodings', help="Location to store encoded links")    
    parser.add_argument('--load_encodings', action='store_true', default=False, help='Use if KB and links are already encoded and can be loaded')

    parser.add_argument('--input_file', help='Location of input test file')
    parser.add_argument('--topk', type=int, default=30, help='Number of candidates to retrieve')
    parser.add_argument('--testing', action='store_true', default=False, help='Use if input file has gold standard KB entry ID; computes recall')
    parser.add_argument('--exact', action='store_true', default=False, help='Use for exact match decoding before pivoting (recommended)')
    parser.add_argument('--nopivot_decode', action='store_true', default=False, help='Use for decoding without pivoting')
    parser.add_argument('--pivot_decode', action='store_true', default=False, help='Use for decoding with pivoting')
    parser.add_argument('--outfile', help='Filename for recall and candidate output files')
    args, unknown = parser.parse_known_args()

    training_data = TrainDataLoader(filename=args.train_file,panphon=args.panphon)
    model = MaxMarginEncoder(embed_size=args.embed_size, hidden_size=args.hidden_size, panphon=args.panphon, model_name=args.model_file, source_vocab_size=training_data.source_vocab_size(), target_vocab_size=training_data.target_vocab_size(), load_model=True)

    pivot_data = PivotDataLoader(kb_filename=args.kb, links_filename=args.links, kb_encoding_path=args.kb_encodings, links_encoding_path=args.links_encodings, encoder_model=model, load_encodings=args.load_encodings, training_data_loader=training_data)

    # Do not set if only encoding
    if args.nopivot_decode or args.pivot_decode:
        if args.exact:
            exactdecoder = ExactDecode(data_loader=ExactDataLoader(kb_filename=args.kb, links_filename=args.links))
        else:
            exactdecoder = None

        pivotdecoder = PivotDecode(data_loader=pivot_data)

        tester = PivotELTest(input_file=args.input_file, encoder_model=model, train_data_loader=training_data, pivotdecoder=pivotdecoder, exactdecoder=exactdecoder, testing=args.testing)

        if args.nopivot_decode:
            preds = tester.test_pivot(pivot=False, topk=args.topk)
            if args.outfile:
                write_topk(filepointer=open(args.outfile + NOPIVOT_OUT, 'w'), preds=preds)

        if args.pivot_decode:
            preds = tester.test_pivot(pivot=True, topk=args.topk)
            if args.outfile:
                write_topk(filepointer=open(args.outfile + PIVOT_OUT, 'w'), preds=preds)
