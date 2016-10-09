import scipy.io
import numpy
import re
import os
import pandas as pd
import subprocess
import time

from datetime import datetime


def generate_test_id(params=None):
    '''This function creates the id for a new test.'''
    now = datetime.now()
    name = 'test'

    if params is not None and 'name' in params:
        name = params['name']

    return '%d-%d-%d-%d-%s' % (now.year, now.month, now.day, time.time(), name)

def load_bin_vec(fname, words):
    '''Loads 300x1 word vecs from Google (Mikolov) word2vec.'''

    vocab = set(words)
    word_vecs = {}

    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = numpy.dtype('float32').itemsize * layer1_size
        print('vocab_size, layer1_size', vocab_size, layer1_size)
        count = 0
        
        for i, line in enumerate(xrange(vocab_size)):
            if i % 1000 == 0:
                print('.',)
            
            word = []
            
            while True:
                ch = f.read(1)

            if ch == ' ' or ch == '':
                print(i, word)
                word = ''.join(word)
                break

            if ch != '\n':
                word.append(ch)

            if word in vocab:
                count += 1
                word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

        print("done")
        print("Words found in wor2vec embeddings", count)
        return word_vecs


def load_glove_vec(fname,words,delimiter,dim):
    vocab = set(words)
    word_vecs = {}

    with open(fname) as f:
        count = 0

        for line in f:
            if line == '':
                continue

            splits = line.replace('\n','').split(delimiter)
            word = splits[0]

            if (word in vocab) or (word.lower() in vocab) or len(vocab) == 0:
                count += 1
                word_vecs[word] = numpy.asarray(splits[1:dim+1],dtype='float32')
                
                if count % 100000 == 0:
                    print('Word2Vec count: ', count)

    return word_vecs
