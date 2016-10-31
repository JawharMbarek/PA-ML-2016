#!/usr/bin/env python

import sys
import pickle

from gensim.models import Word2Vec

if len(sys.argv) != 3:
    print('ERROR: To much/few arguments!')
    print('       python scripts/extract_embeddings_vocabulary.py <embeddings> <vocab out>')
    sys.exit(2)

argv = sys.argv[1:]

emb_path = argv[0]
out_path = argv[1]
word2vec = Word2Vec.load(emb_path)

new_vocab = {}

for k, v in word2vec.vocab.items():
    new_vocab[k] = v.index

with open(out_path, 'wb') as f:
    pickle.dump(new_vocab, f)

print('Successfully extracted the vocabulary of the embeddings %s' % emb_path)