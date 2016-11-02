from tsv_data_loader import TsvDataLoader

import os
import sys
import pickle
import numpy as np

argv = sys.argv[1:]

vocab = pickle.load(argv[0])
loader = TsvDataLoader(argv[1], vocab)

X_train = np.array()
Y_train = np.array()

for X, Y in loader.load_lazy():
    X_train.append(X)
    Y_train.append(Y)

    if len(X_train) % 10000 == 0:
        print('Loaded %d reviews')

X_train.save('x_train_amazon_distant.npy')
Y_train.save('y_train_amazon_distant.npy')
