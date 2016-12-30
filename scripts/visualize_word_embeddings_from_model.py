#!/usr/bin/env python
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import operator
import pickle
import h5py

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from os import path

list_of_words = None
model_weights_path = None
vocabulary_path = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:w:v:',
                               ['embeddings=', 'list_of_words=', 'vocabulary='])
except getopt.GetoptError as e:
    print('./visualize_word_embeddings_from_model.py -m <model-weights> -w <list-of-words> -v <vocabulary>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-w', '--list-of-words'):
        list_of_words = arg.split(',')
    elif opt in ('-m', '--model-weights'):
        model_weights_path = arg
    elif opt in ('-v', '--vocabulary'):
        vocabulary_path = arg

if list_of_words is None or model_weights_path is None or vocabulary_path is None:
    print('ERROR: Missing mandatory argument(s)')
    print('       python visualize_word_embeddings_from_model.py -m <model-weights> -w <list-of-words> -v <vocabulary>')
    sys.exit(2)

print('Loading embeddings for given words...')

if len(list_of_words) == 1 and path.isfile(list_of_words[0]):
    with open(list_of_words[0], 'r') as f:
        list_of_words = f.read().split('\n')

vocabulary = None

with open(vocabulary_path, 'rb') as f:
    vocabulary = pickle.load(f)

emb_weights = None
word_vectors = {}

with h5py.File(model_weights_path) as weights_f:
    group_keys = list(weights_f['model_weights'].keys())
    emb_key = list(filter(lambda x: x.startswith('embedding_'), group_keys))

    if len(emb_key) > 1:
        print('ERROR: Ambigious embedding layers found! Exiting...')
        sys.exit(2)
    else:
        emb_key = emb_key[0]

    emb_weights = weights_f['model_weights'][emb_key]
    emb_weights = emb_weights[list(emb_weights.keys())[0]]

    for word in list_of_words:
        if word in vocabulary:
            word_vectors[word] = emb_weights[vocabulary[word]]
        else:
            print('ERROR: Word "%s" missing in word embeddings! Exiting...' % word)
            sys.exit(2)

print('Loaded embeddings for given words!')

only_word_vectors = np.array(list(word_vectors.values()))

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)

pca = PCA(n_components=2)
Y_pca = pca.fit(only_word_vectors).transform(only_word_vectors)
Y = tsne.fit_transform(only_word_vectors)

import pdb
pdb.set_trace()

plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 2.0)
plt.scatter(Y_pca[:, 0], Y_pca[:, 1])

for label, x, y in zip(word_vectors.keys(), Y_pca[:, 0], Y_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.show() 
