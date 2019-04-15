"""Merging script for entity linking candidate lists.

Use to merge candidate files, if pivot-based-entity-linking is parallelized across different sections of the KB. Useful if KB is very large and/or many cores are available for PBEL computation.

Author: Shruti Rijhwani (srijhwan@cs.cmu.edu)
Last update: 2019-04-15
"""

import argparse
import codecs
import glob
from collections import defaultdict
import codecs
from utils.constants import PIVOT_OUT,DELIM,RECALL_OUT,CAND_OUT,ID_IDX

def get_scores(prefix):
    files = glob.glob(prefix + "*" + PIVOT_OUT)
    scores = defaultdict(lambda: defaultdict(lambda: 0))

    for filen in files:
        ent_count = 0
        
        with open(filen, 'r') as f:
            for line in f:
                if '***' in line:
                    ent_count += 1
                    continue
                spl = line.strip().split(DELIM)
                scores[ent_count][int(spl[0])] = float(spl[1])
    return scores

def read_test(filename):
    data = []
    with codecs.open(filename, 'r', 'utf8') as f:
        for line in f:
            spl = line.strip().split(DELIM)
            data.append(int(spl[ID_IDX]))
    return data

def get_recall(test_data, scores, n):
    assert len(test_data) == len(scores)
    recall = 0.0
    for idx, id_num in enumerate(test_data):
        cur_scores = scores[idx]
        sorted_scores = sorted(cur_scores, key=cur_scores.get, reverse=True)
        if id_num in sorted_scores[:n]:
            recall += 1
    return recall/len(test_data)

def get_candidates(test_data, scores, n):
    assert len(test_data) == len(scores)
    candidates = []
    for idx, name in enumerate(test_data):
        cur_scores = scores[idx]
        sorted_scores = sorted(cur_scores, key=cur_scores.get, reverse=True)
        candidates.append((name, sorted_scores[:n]))
    return candidates  

parser = argparse.ArgumentParser()
parser.add_argument('--prefix')
parser.add_argument('--test_file')
parser.add_argument('--n', type=int, default=30)
parser.add_argument('--recall', action='store_true')
parser.add_argument('--candidates', action='store_true')
args = parser.parse_args()

test_data = read_test(args.test_file)
scores = get_scores(args.prefix)

if args.recall:
    recall = get_recall(test_data, scores, args.n)
    with open(args.prefix + PIVOT_OUT + RECALL_OUT, 'w') as f:
        f.write(str(recall) + '\n')
elif args.candidates:
    candidates = get_candidates(test_data, scores, args.n)
    with codecs.open(args.prefix + PIVOT_OUT + CAND_OUT, 'w', 'utf8') as f:
        for idx, (name,cands) in enumerate(candidates):
            f.write(name + DELIM)
            for cand in cands[:-1]:
                f.write(str(cand) + ' | ' + str(scores[idx][cand]) + ' || ')
            f.write(str(cands[-1]) + ' | ' + str(scores[idx][cands[-1]]))
            f.write('\n')
