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
import time

from keras.models import Sequential
from keras.layers import Merge, Dense, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from data_utils import compute_class_weights
from data_loader import DataLoader
from evaluation_metrics import f1_score_pos_neg
from model import Model

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Missing mandatory argument')
    print('       python scripts/meta_classifier_evaluation.py <opt-tsv> <test-tsv>')
    sys.exit(2)

SENTENCE_LENGTH = 140
MODELS_PATH = path.abspath(path.join(path.dirname(__file__), '..', 'models'))
VOCABS_PATH = path.abspath(path.join(path.dirname(__file__), '..', 'vocabularies'))

test_data_path = argv[0]
val_data_path = argv[1]

def get_domain_model_dir(d):
    return path.join(MODELS_PATH, 'best_model_crossdomain_we_ds_%s' % d)

domains = ['dai', 'dil', 'hul', 'mpq', 'semeval', 'tac']
vocabs  = ['vocab_en300M_reduced.pickle', 'vocab_news_emb.pickle', 'vocab_wiki_emb.pickle']

models = {}
vocab_per_length = {}
vocab_per_domain = {}

val_data_per_vocab = {}
data_per_vocab = {}

val_data_per_domain = {}
data_per_domain = {}

y_true = None
y_val_true = None

timestamp = int(time.time())
model_checkpoint_path = 'expert_net_weights_%d.h5' % timestamp
model_json_path = 'experts_net_models_%d.json' % timestamp

if path.isfile(model_checkpoint_path):
    os.remove(model_checkpoint_path)

if path.isfile(model_json_path):
    os.remove(model_json_path)

print('Loading models...')

for i, domain in enumerate(domains):
    model_dir = get_domain_model_dir(domain)
    model_json_file = path.join(model_dir, 'model.json')
    model_weights_file = path.join(model_dir, 'weights_1.h5')

    with open(model_json_file, 'r') as f:
        models[domain] = keras.models.model_from_json(f.read())

    models[domain].load_weights(model_weights_file)

    # rename layers to mitigate the unique name problem
    # see https://github.com/fchollet/keras/issues/3974
    for j in range(len(models[domain].layers)):
        models[domain].layers[j].name += '_%s_model' % domain

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
            y_val_true = val_sents

print('Loaded vocabularies and data!')
print('Starting to preprocess the data for the combined network...')

trained_models = []
data_shape = list(data_per_vocab.values())[0].shape

combined_x = []
combined_val_x = []

for j, (domain, model) in enumerate(models.items()):
    trained_models.append(model)

    vocab_len = model.layers[0].input_dim - 1
    vocab_per_domain[domain] = vocab_per_length[vocab_len]

    data_per_domain[domain] = data_per_vocab[vocab_len]
    val_data_per_domain[domain] = val_data_per_vocab[vocab_len]

    combined_x.append(data_per_domain[domain])
    combined_val_x.append(val_data_per_domain[domain])

print('Finished preprocessing the data for the combined network!')
print('Starting to assemble and optimize the meta-classifier...')

expert_net = Sequential()
expert_net.add(Merge(trained_models, mode='concat'))
expert_net.add(Dense(256))
expert_net.add(Activation('relu'))
expert_net.add(Dense(3, activation='softmax'))
expert_net = Model.compile(expert_net)

class_weights = compute_class_weights(y_true)
validation_data = (combined_val_x, to_categorical(y_val_true, 3))
early_stopping = EarlyStopping(patience=50, verbose=1, mode='max', monitor='val_f1_score_pos_neg')
model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_path, mode='max', save_best_only=True,
                                   monitor='val_f1_score_pos_neg')

expert_net.fit(combined_x, to_categorical(y_true),
               validation_data=validation_data,
               nb_epoch=1000, batch_size=500,
               callbacks=[early_stopping, model_checkpoint])

with open(model_json_path, 'w+') as f:
    f.write(expert_net.to_json())

expert_net.load_weights(model_checkpoint_path)
y_pred = expert_net.predict(combined_val_x)

y_true = K.variable(value=to_categorical(y_val_true, 3))
y_pred = K.variable(value=y_pred)

res_pos_neg = K.eval(f1_score_pos_neg(y_true, y_pred))

print('Finished assembling and optimizing the meta-classifier!')
print('The achieved f1_score_pos_neg is: %f' % res_pos_neg)
