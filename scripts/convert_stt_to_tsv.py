#!/usr/bin/env python

import os
import sys
import shutil

from os import path

if len(sys.argv) != 5:
    print('ERROR: python scripts/convert_stt_to_tsv.py <sentences-txt>' +
          ' <sents-txt> <split-txt> <out-tsv>')
    sys.exit(2)

sentences_path, sentiments_path, split_path, out_path  = sys.argv[1:]

sentences = {}
sentiments = {}
splits = {}

with open(sentences_path, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue

        tid, text = line.split(' ', 1)
        sentences[tid] = text

    print('Read all sentences... (total %d)' % len(sentences))

with open(sentences_path, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue

        

    print('Read all splits... (total %d)' % len(sentences))

with open(sentiments_path, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue

        tid, sent = line.split('|')

        sent = float(sent)

        if sent > 0.6:
            sent = 'positive'
        elif sent < 4.0:
            sent = 'negative'
        else:
            sent = 'neutral'

        sentiments[tid] = sent

    print('Read all sentiments... (total %d)' % len(sentiments))
