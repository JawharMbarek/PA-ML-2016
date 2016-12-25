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
from keras.layers import Merge, Dense, Activation
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
print('Starting to preprocess the data for the combined network...')

trained_models = []
data_shape = list(data_per_vocab.values())[0].shape

combined_x = []
combined_val_x = []

for domain, model in models.items():
    trained_models.append(model)

    vocab_len = model.layers[0].input_dim - 1
    vocab_per_domain[domain] = vocab_per_length[vocab_len]
    data_per_domain[domain] = data_per_vocab[vocab_len]

    for i in range(data_shape[0]):
        curr_x = []
        curr_val_x = []

        domain_data_x = np.array(data_per_domain[domain][i])
        domain_val_data_x = np.array(data_per_domain[domain][i])

        if i < len(combined_x):
            curr_x = combined_x[i]
            curr_val_x = combined_val_x[i]
        else:
            combined_x.append([])
            combined_val_x.append([])

        curr_x.append(domain_data_x.reshape(1, 140))
        curr_val_x.append(domain_val_data_x.reshape(1, 140))

        combined_x[i] = curr_x
        combined_val_x[i] = curr_val_x

print('Finished preprocessing the data for the combined network!')
print('Starting to assemble and optimize the meta-classifier...')

keras_models = list(models.values())

expert_net = Sequential()
expert_net.add(Merge(trained_models, mode='concat'))
expert_net.add(Dense(24))
expert_net.add(Activation('relu'))
expert_net.add(Dense(3, activation='softmax'))
expert_net = Model.compile(expert_net)

import pdb
pdb.set_trace()

y_true = K.variable(value=to_categorical(y_true, 3))
y_pred = K.variable(value=y_pred)

res_pos_neg = K.eval(f1_score_pos_neg(y_true, y_pred))

print('Finished assembling and optimizing the meta-classifier!')
print('The final weights are: %s' % str(weights_per_domain))
print('The achieved f1_score_pos_neg is: %f' % res_pos_neg)
