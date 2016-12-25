#!/usr/bin/env python

import sys
import os
from os import path

# Hack to be able to import DataLoader / parse_utils
dlp = path.realpath(path.join(path.dirname(__file__), '..', 'source'))
sys.path.insert(0, dlp)

import keras
import pickle
import numpy as np
import keras.backend as K

from keras.models import Sequential
from keras.layers import Merge, Dense
from keras.utils.np_utils import to_categorical
from data_loader import DataLoader
from evaluation_metrics import f1_score_pos_neg
from model import Model

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Missing mandatory argument')
    print('       python scripts/meta_classifier_evaluation.py <opt-tsv> <test-tsv>')
    sys.exit(2)

MODELS_PATH = path.abspath(path.join(path.dirname(__file__), '..', 'models'))
VOCABS_PATH = path.abspath(path.join(path.dirname(__file__), '..', 'vocabularies'))

test_data_path = argv[0]
val_data_path = argv[1]

def get_domain_model_dir(d):
    return path.join(MODELS_PATH, 'best_model_crossdomain_we_ds_%s' % d)

domains = ['dai', 'dil', 'hul', 'jcr', 'mpq', 'sem', 'semeval', 'tac']
vocabs  = ['vocab_en300M_reduced.pickle', 'vocab_news_emb.pickle', 'vocab_wiki_emb.pickle']

models = {}
vocab_per_length = {}
vocab_per_domain = {}
weights_per_domain = {}
val_data_per_vocab = {}
data_per_vocab = {}
data_per_domain = {}
y_true = None

print('Loading models...')

for domain in domains:
    model_dir = get_domain_model_dir(domain)
    model_json_file = path.join(model_dir, 'model.json')
    model_weights_file = path.join(model_dir, 'weights_1.h5')

    with open(model_json_file, 'r') as f:
        models[domain] = keras.models.model_from_json(f.read())

    models[domain].load_weights(model_weights_file)

print('Models loaded!')
print('Loading vocabularies and data...')

for vocab in vocabs:
    vocab_path = path.join(VOCABS_PATH, vocab)
    vocab_length = 0
    vocab_dict = None

    with open(vocab_path, 'rb') as f:
        vocab_dict = pickle.load(f)
        vocab_length = len(vocab_dict)
        vocab_per_length[vocab_length] = vocab_dict

    if vocab_length not in data_per_vocab:
        sents, txts, _, _ = DataLoader.load(test_data_path, vocab_dict, randomize=True)
        val_sents, val_txts, _, _ = DataLoader.load(val_data_path, vocab_dict, randomize=True)
        
        data_per_vocab[vocab_length] = txts
        val_data_per_vocab[vocab_length] = val_txts

        if y_true is None:
            y_true = sents

print('Loaded vocabularies and data!')
print('Starting to assemble and optimize the meta-classifier...')

def transform_data_per_domain():
    pass

keras_models = list(models.values())
expert_net = Sequential()
expert_net.add(Merge(keras_models, mode='concat'))
expert_net.add(Dense(3))

import pdb
pdb.set_trace()

for domain, model in models.items():
    weights_per_domain[domain] = 1.0 / len(models)
    vocab_len = model.layers[0].input_dim - 1
    vocab_per_domain[domain] = vocab_per_length[vocab_len]
    data_per_domain[domain] = data_per_vocab[vocab_len]

data_length = len(list(data_per_domain.values())[0])
pred_per_domain = {}
y_pred = np.zeros((data_length, 3))

loss_per_epoch = []
weights_per_epoch = []

for domain in models.keys():
    x = data_per_domain[domain]
    pred_per_domain[domain] = model.predict(x)

    domain_pred = pred_per_domain[domain]

    import pdb
    pdb.set_trace()

    error = domain_pred - to_categorical(y_true, 3)
    loss = np.sum(error ** 2)
    gradient = x.T.dot(error) / x.shape[0]

    weights_per_domain[domain] = -0.01 * gradient

    loss_per_epoch.append(loss)

    y_pred[:,:] += weights_per_domain[domain] * pred_per_domain[domain]

y_true = K.variable(value=to_categorical(y_true, 3))
y_pred = K.variable(value=y_pred)

res_pos_neg = K.eval(f1_score_pos_neg(y_true, y_pred))

print('Finished assembling and optimizing the meta-classifier!')
print('The final weights are: %s' % str(weights_per_domain))
print('The achieved f1_score_pos_neg is: %f' % res_pos_neg)
