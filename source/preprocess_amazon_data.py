import os
import sys
import pickle
import numpy as np
import gzip
import re
import parse_utils
import time
import h5py

from nltk import TweetTokenizer
from keras.utils.np_utils import to_categorical

SENTIMENT_MAP = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

argv = sys.argv[1:]

vocab = pickle.load(open(argv[0], 'rb'))
dummy_word_idx = vocab.get('DUMM_WORD_IDX', 1)

tmp_x = []
tmp_y = []

store_size = 100000
sent_length = 140
counter = 0

review_count = int(argv[2])

tknzr = TweetTokenizer(reduce_len=3)

output_file = 'preprocessed/amazon_distant_train_preprocessed.hdf5'

with open(argv[1], 'r') as tsvf:
    with h5py.File(output_file) as f:
        x_dataset = f.create_dataset('x', dtype='i8', shape=(review_count, sent_length))
        y_dataset = f.create_dataset('y', dtype='i8', shape=(review_count, 3))

        start_time = time.time()

        for i in range(0, review_count):
            try:
                curr_line = tsvf.readline().split('\t')

                curr_text = curr_line[-1]
                curr_text = parse_utils.preprocess_tweet(curr_text.replace('\n', '').lower())
                curr_text = tknzr.tokenize(curr_text)

                curr_sent = SENTIMENT_MAP[curr_line[-2]]

                tmp_x.append(curr_text)
                tmp_y.append(curr_sent)

                counter += 1

                if len(tmp_x) % store_size == 0:
                    x_dataset[i-store_size+1:i+1] = parse_utils.convert2indices(tmp_x, vocab, dummy_word_idx, max_sent_length=sent_length)
                    y_dataset[i-store_size+1:i+1] = np.array(to_categorical(tmp_y), dtype=np.int32)

                    tmp_x = []
                    tmp_y = []

                    print('Processed and saved %d reviews (took %fs)' % (counter, time.time() - start_time))
                    
                    start_time = time.time()
            except EOFError:
                break
