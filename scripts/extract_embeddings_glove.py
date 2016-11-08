#!/usr/bin/env python

import _pickle as cPickle
import os
import numpy as np
import getopt
import sys
import time

from utils import load_glove_vec


def main(argv):
    np.random.seed(int(time.time()))
    data_dir = 'embeddings'
    vocab_dir = 'vocabularies'
    emb_path = ''
    emb_name = ''
    fname_vocab = ''
    out_dir = ''

    try:
        opts, args = getopt.getopt(argv, "v:e:o:", ['out=', "vocab=", "embedding="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-v", "--vocab"):
            fname_vocab = arg
        elif opt in ("-e", "--embedding"):
            emb_path = arg
            emb_name = arg
        elif opt in ('-o', '--output'):
            out_dir = arg

    # get vocabulary
    print(fname_vocab)
    alph = cPickle.load(open(fname_vocab, 'rb'))
    words = alph.keys()
    print("Vocab size", len(alph))

    word2vec = {}

    # get embeddings
    fname, delimiter, ndim = (emb_path, ' ', 52)
    word2vec.update(load_glove_vec(fname, words, delimiter, ndim))

    print(len(word2vec.keys()))
    ndim = len(word2vec[list(word2vec.keys())[0]])
    print('ndim', ndim)

    random_words_count = 0
    vocab_emb = np.zeros((len(alph) + 1, ndim), dtype='float32')

    for word, idx in alph.items():
        word_vec = word2vec.get(word, None)
        if word_vec is None or word_vec.shape[0] != 52:
            word_vec = np.random.uniform(-0.25, 0.25, ndim)
            random_words_count += 1

        vocab_emb[idx] = word_vec

    print('random_words_count', random_words_count)
    print(vocab_emb.shape)

    outfile = '{}_emb.npy'.format(emb_name)
    print(outfile)

    np.save(outfile, vocab_emb)


if __name__ == '__main__':
    main(sys.argv[1:])