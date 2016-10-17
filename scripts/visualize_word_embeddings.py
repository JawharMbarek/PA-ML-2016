#!/usr/bin/env python
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec 
from sklearn.manifold import TSNE

embeddings_file = ''
nb_words = 1000

try:
    opts, args = getopt.getopt(sys.argv[1:], 'e:n:',
                               ['embeddings=', 'nr_of_words='])
except getopt.GetoptError:
    print('./visualize_word_embeddings.py -e <embeddings> -n <nr-of-words>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-e', '--embeddings'):
        embeddings_file = arg
    elif opt in ('-n', '--nr-of-words'):
        nb_words = arg
 
print('Loading embeddings...')
 
word_vectors = {}
word_embeddings = Word2Vec.load(embeddings_file)
counter = 0

for word in word_embeddings.vocab.keys():
    word_vectors[word] = word_embeddings[word]
    counter += 1

    if counter % 10000 == 0:
        print('Loaded %d words and their embeddings...' % counter)

print('Loaded embeddings')

only_word_vectory = np.array(list(word_vectors.values()))

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(only_word_vectory[:nb_words,:])

plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(word_vectors.keys(), Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.show() 
