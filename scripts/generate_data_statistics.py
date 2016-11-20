#!/usr/bin/env python

import sys
import time
import json
import getopt
import pickle

from os import path

# Hack to be able to import DataLoader / parse_utils
dlp = path.realpath(path.join(path.dirname(__file__), '..', 'source'))
sys.path.insert(0, dlp)

import parse_utils

from data_loader import DataLoader
from nltk import TweetTokenizer

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: voacabulary or data files missing!')
    print('       python scripts/generate_data_scripts <vocabulary> <data files>')
    sys.exit(2)

vocabulary_path = argv[0]
data_paths = argv[1:]

vocabulary = None
stats_file = 'text_statistics_%d.json' % int(time.time())
tokenizer = TweetTokenizer(reduce_len=True)

stats = {'data': {}}
stats_vocab = {}

with open(vocabulary_path, 'rb') as f:
    vocabulary = pickle.load(f)

for file_path in data_paths:
    if file_path.endswith('.tsv'):
        sentiments, _, data, _ = DataLoader.load(file_path, vocabulary)

        print('Starting to process file %s' % file_path)

        sentence_count = 0
        sentence_word_len = 0
        sentence_char_len = 0

        words_missing_in_vocab = 0
        words_present_in_vocab = 0

        sentiment_pos_count = sum([x == 2 for x in sentiments])
        sentiment_neu_count = sum([x == 1 for x in sentiments])
        sentiment_neg_count = sum([x == 0 for x in sentiments])

        for t in map(lambda x: x[-1], data):
            words = tokenizer.tokenize(parse_utils.preprocess_tweet(t))

            for w in words:
                if vocabulary.get(w) is None:
                    words_missing_in_vocab += 1
                else:
                    words_present_in_vocab += 1

                if w not in stats_vocab:
                    stats_vocab[w] = 1
                else:
                    stats_vocab[w] += 1

            sentence_count += 1

            sentence_word_len += len(words)
            sentence_char_len += sum([len(w) for w in words])

            if sentence_count % 1000 == 0:
                print('Processed %d sentences' % sentence_count)

        if sentence_count == 0:
            continue

        stats['data'].setdefault(file_path, {
            'total': {
                'word_count': int(sentence_word_len),
                'sentence_count': int(sentence_count),
                'character_count': int(sentence_char_len),
                'sentiment_pos': int(sentiment_pos_count),
                'sentiment_neg': int(sentiment_neg_count),
                'sentiment_neu': int(sentiment_neu_count),
                'words_missing_in_vocab': int(words_missing_in_vocab),
                'words_present_in_vocab': int(words_present_in_vocab),
                'words_missing_in_vocab_percentage': float(words_missing_in_vocab / float(sentence_word_len)),
                'words_precent_in_vocab_percentage': float(words_present_in_vocab / float(sentence_word_len))
            },
            'avg': {
                'word_count': float(sentence_word_len / float(sentence_count)),
                'character_count': float(sentence_char_len / float(sentence_count)),
                'sentiment_pos_percentage': float(sentiment_pos_count / float(sentence_count)),
                'sentiment_neg_percentage': float(sentiment_neg_count / float(sentence_count)),
                'sentiment_neu_percentage': float(sentiment_neu_count / float(sentence_count)),
                'words_missing_in_vocab': float(words_missing_in_vocab / float(sentence_count)),
                'words_present_in_vocab': float(words_present_in_vocab / float(sentence_count))
            }
        })

        print('Finished analyzing the file %s' % file_path)
    else:
        print('Cannot analyse non-TSV file %s' % file_path)

stats['vocabulary'] = stats_vocab

with open(stats_file, 'w+') as f:
    f.write(json.dumps(stats))

print('Successfully saved text statistics to %s' % stats_file)
