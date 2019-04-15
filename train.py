"""Trains a entity similarity model between two languages using a character-level encoder.

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
import panphon as pp
import logging
from max_margin_encoder import MaxMarginEncoder
from traindataloader import TrainDataLoader
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class ELTrainer():
    def __init__(self, model, training_data, val_data):
        self.model = model
        self.training_data = training_data.data
        self.validation_data = val_data

    def get_val_recall(self, encode_batch_size=512):
        recall = 0.0
        total = 0

        source_reps = []
        target_reps = []

        for i in range(0, len(self.validation_data), encode_batch_size):
            cur_size = min(encode_batch_size, len(self.validation_data) - i)
            batch = self.validation_data[i:i + cur_size]
            source_reps += self.model.encode_source([s for s, t in batch])
            target_reps += self.model.encode_target([t for s, t in batch])

        # get reps for whole training data as validation KB
        for i in range(0, len(self.training_data), encode_batch_size):
            cur_size = min(encode_batch_size, len(self.training_data) - i)
            batch = self.training_data[i:i + cur_size]
            source_reps += self.model.encode_source([s for s, t in batch])

        scores = np.array(target_reps).dot(np.array(source_reps).T)

        for entry_idx, entry_scores in enumerate(scores):
            ranks = entry_scores.argsort()[::-1]
            if ranks[0] == entry_idx:
                recall += 1
            total += 1

        return recall / total
    
    def train(self, epochs, trainer, batch_size=64, patience=50, early_stop_check=5):
        if trainer == 'sgd':
            trainer = dy.SimpleSGDTrainer(self.model.model)
        elif trainer == 'adam':
            trainer = dy.AdamTrainer(self.model.model)

        best = 0
        last_updated = 0

        for ep in range(epochs):
            ep_loss = 0
            random.shuffle(self.training_data)
            logging.info("Epoch: %d" % ep)
            for i in range(0, len(self.training_data), batch_size):
                cur_size = min(batch_size, len(self.training_data) - i)
                loss = self.model.calculate_loss(self.training_data[i:i + cur_size])
                ep_loss += loss.scalar_value()
                loss.backward()
                trainer.update()

            logging.info("Train loss: %0.4f" % (ep_loss / len(self.training_data)))

            if ep % early_stop_check == 0:
                recall = self.get_val_recall()
                if recall > best:
                    best = recall
                    last_updated = ep
                    logging.info('Saved: %0.4f' % best)
                    self.model.save_model()

            if ep - last_updated > patience:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='Number of iterations through training data')
    parser.add_argument('--trainer', default='sgd', help='Training algorithm to use; options are "sgd" or "adam" (add code for any other Dynet trainer)')
    parser.add_argument('--embed_size', type=int, default=64, help='Character embedding size')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Character LSTM hidden size')
    parser.add_argument('--train_file', help='Location of training file with English-HRL links')
    parser.add_argument('--val_file', help='Location of validation file with English-HRL or English-LRL links')
    parser.add_argument('--model_file', help='Location to store best model')
    parser.add_argument('--finetune', action='store_true', default=False, help='Reload saved model and finetune')
    parser.add_argument('--panphon', action='store_true', default=False, help='Use PanPhon embeddings instead of IPA embeddings (IPA input only)')
    args, unknown = parser.parse_known_args()

    training_data = TrainDataLoader(filename=args.train_file, panphon=args.panphon)
    val_data = training_data.convert_test_data(args.val_file)

    model = MaxMarginEncoder(embed_size=args.embed_size, hidden_size=args.hidden_size, panphon=args.panphon, model_name=args.model_file, load_model=args.finetune, source_vocab_size=training_data.source_vocab_size(), target_vocab_size=training_data.target_vocab_size())

    trainer = ELTrainer(model=model, training_data=training_data, val_data=val_data)
    trainer.train(args.epochs, args.trainer)
