import os
import sys
import pickle
import numpy as np
import gzip
import re
import time
import h5py
import getopt

from os import path
from nltk import TweetTokenizer
from keras.utils.np_utils import to_categorical

# Hack to be able to import DataLoader / parse_utils
dlp = path.realpath(path.join(path.dirname(__file__), '..', 'source'))
sys.path.insert(0, dlp)

import parse_utils

SENTIMENT_MAP = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

argv = sys.argv[1:]

sent_length = -1
max_count = -1
output_path = None
tsv_path = None
vocab_path = None

try:
    opts, args = getopt.getopt(argv, 'o:t:s:v:m:',
                               ['output=', 'tsv=', 'sent_length=', 'vocab=', 'max_count='])
except getopt.GetoptError:
    print('./preprocess_tsv_data.py -t <tsv> -o <output> -s <sent-length> -v <vocab> -m <max-count>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-t', '--tsv'):
        tsv_path = arg
    elif opt in ('-o', '--output'):
        output_path = arg
    elif opt in ('-s', '--sent-length'):
        sent_length = int(arg)
    elif opt in ('-v', '--vocab'):
        vocab_path = arg
    elif opt in ('-m', '--max-count'):
        max_count = int(arg)

if sent_length == -1 or output_path is None or tsv_path is None or vocab_path is None or max_count == -1:
    print('ERROR: Missing mandatory parameters!')
    print('./preprocess_tsv_data.py -t <tsv> -o <output> -s <sent-length> -v <vocab> -m <max-count>')
    sys.exit(2)

vocab = pickle.load(open(vocab_path, 'rb'))
dummy_word_idx = vocab.get('DUMM_WORD_IDX', 1)

tmp_x = []
tmp_y = []

counter = 0
store_size = 1000000
tknzr = TweetTokenizer(reduce_len=3)

with open(tsv_path, 'r') as tsvf:
    with h5py.File(output_path) as f:
        x_dataset = f.create_dataset('x', dtype='i8', shape=(max_count, sent_length))
        y_dataset = f.create_dataset('y', dtype='i8', shape=(max_count, 3))

        start_time = time.time()

        curr_idx = 0
        running = True

        while (curr_idx + 1) < max_count and running:
            if store_size > max_count:
                store_size = max_count

            for i in range(0, store_size):
                curr_line = tsvf.readline().split('\t')

                if len(curr_line) == 1 and curr_line[0] == '':
                    running = False
                    break

                curr_text = curr_line[-1]
                curr_text = parse_utils.preprocess_tweet(curr_text.replace('\n', '').lower())
                curr_text = tknzr.tokenize(curr_text)
                curr_sent = SENTIMENT_MAP[curr_line[-2]]

                tmp_x.append(curr_text)
                tmp_y.append(curr_sent)

            if len(tmp_x) > 0 and len(tmp_y) > 0:
                x_dataset[curr_idx:curr_idx+len(tmp_x)] = parse_utils.convert2indices(tmp_x, vocab, dummy_word_idx, max_sent_length=sent_length)
                y_dataset[curr_idx:curr_idx+len(tmp_y)] = np.array(to_categorical(tmp_y, nb_classes=3), dtype=np.int32)

                curr_idx += len(tmp_x)

                print('Processed and saved %d/%d training examples (took %fs)' % (curr_idx, max_count, time.time() - start_time))
            
            tmp_x = []
            tmp_y = []

            start_time = time.time()

print('Finished preprocessing %s' % tsv_path)
print('Stored preprocessed data in %s' % output_path)
