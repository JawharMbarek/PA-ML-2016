#!/usr/bin/env python

import _pickle as cPickle
import os
import numpy as np
import getopt
import sys

from gensim.models import Word2Vec


def main(argv):
    np.random.seed(123)
    data_dir = 'embeddings'
    vocab_dir = 'vocabularies'
    emb_path = ''
    emb_name = ''
    fname_vocab = ''
    outfile = ''

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
            outfile = arg

    # get vocabulary
    print(fname_vocab)
    vocab = cPickle.load(open(fname_vocab, 'rb'))
    words = vocab.keys()
    print('The vocabulary size is: %d' % len(vocab))

    word2vec = Word2Vec.load(emb_path)
    emb_dim = len(word2vec[list(word2vec.vocab.keys())[0]])

    print('Size of the embeddings vocabulary: %d' % len(word2vec.vocab))
    print('Dimensionality of the embeddings: %d' % emb_dim)

    random_words_count = 0
    vocab_emb = np.zeros((len(vocab) + 1, emb_dim), dtype='float32')

    for word, idx in vocab.items():
        word_vec = None

        if word not in word2vec or word2vec[word].shape[0] != 52:
            word_vec = np.random.uniform(-0.25, 0.25, emb_dim)
            random_words_count += 1
        else:
            word_vec = word2vec[word]

        vocab_emb[idx] = word_vec

    print('Words with random embeddings: %d' % random_words_count)
    print('Shape of the embeddings matrix: %s' % str(vocab_emb.shape))
    print('Saving embedding matrix to: %s' % outfile)

    np.save(outfile, vocab_emb)


if __name__ == '__main__':
    main(sys.argv[1:])
