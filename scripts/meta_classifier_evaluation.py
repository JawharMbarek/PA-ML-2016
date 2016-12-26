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
import json

from keras.models import Sequential
from keras.layers import Merge, Dense, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from data_utils import compute_class_weights
from data_loader import DataLoader
from evaluation_metrics import f1_score_pos_neg
from model import Model

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Missing mandatory argument')
    print('       python scripts/meta_classifier_evaluation.py <opt-tsv> <val-tsv> <test-tsv>')
    sys.exit(2)

SENTENCE_LENGTH = 140
ROOT_PATH = path.abspath(path.join(path.dirname(__file__), '..'))
MODELS_PATH = path.join(ROOT_PATH, 'models')
VOCABS_PATH = path.join(ROOT_PATH, 'vocabularies')
META_RESULTS_PATH = path.join(ROOT_PATH, 'meta-results')

if not path.isdir(META_RESULTS_PATH):
    os.mkdir(META_RESULTS_PATH)

train_data_path = argv[0]
test_data_path = argv[1]
val_data_path = argv[2]

def get_domain_model_dir(d):
    return path.join(MODELS_PATH, 'best_model_crossdomain_we_ds_%s' % d)

domains = ['dai', 'dil', 'hul', 'mpq', 'semeval', 'tac']
vocabs  = ['vocab_en300M_reduced.pickle', 'vocab_news_emb.pickle', 'vocab_wiki_emb.pickle']

models = {}
vocab_per_length = {}
vocab_per_domain = {}

train_data_per_vocab = {}
val_data_per_vocab = {}
test_data_per_vocab = {}

train_data_per_domain = {}
val_data_per_domain = {}
test_data_per_domain = {}

y_train_true = None
y_val_true = None
y_test_true = None

target_domain = val_data_path.split('/')[-1].split('_')[0].lower()

timestamp = int(time.time())
current_results_path = path.join(META_RESULTS_PATH, 'run_%s_%d' % (target_domain, timestamp))
model_checkpoint_path = path.join(current_results_path, 'expert_net_weights_%s_%d.h5' % (target_domain, timestamp))
model_json_path = path.join(current_results_path, 'experts_net_models_%s_%d.json' % (target_domain, timestamp))
metrics_json_path = path.join(current_results_path, 'experts_net_metrics_%s_%d.json' % (target_domain, timestamp))
config_json_path = path.join(current_results_path, 'experts_net_config_%s_%d.json' % (target_domain, timestamp))

if not path.isdir(current_results_path):
    os.mkdir(current_results_path)

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

    if vocab_length not in train_data_per_vocab:
        test_sents, test_txts, _, _ = DataLoader.load(test_data_path, vocab_dict, randomize=True)
        val_sents, val_txts, _, _ = DataLoader.load(val_data_path, vocab_dict, randomize=True)
        train_sents, train_txts, _, _ = DataLoader.load(train_data_path, vocab_dict, randomize=True)

        test_data_per_vocab[vocab_length] = test_txts[0:10]
        val_data_per_vocab[vocab_length] = val_txts[0:10]
        train_data_per_vocab[vocab_length] = train_txts[0:10]

        if y_train_true is None:
            y_train_true = train_sents[0:10]
            y_val_true = val_sents[0:10]
            y_train_true = train_sents[0:10]

print('Loaded vocabularies and data!')
print('Starting to preprocess the data for the combined network...')

trained_models = []
data_shape = list(train_data_per_vocab.values())[0].shape

combined_train_x = []
combined_val_x = []
combined_test_x = []

for j, (domain, model) in enumerate(models.items()):
    trained_models.append(model)

    vocab_len = model.layers[0].input_dim - 1
    vocab_per_domain[domain] = vocab_per_length[vocab_len]

    train_data_per_domain[domain] = train_data_per_vocab[vocab_len]
    val_data_per_domain[domain] = val_data_per_vocab[vocab_len]
    test_data_per_domain[domain] = test_data_per_vocab[vocab_len]

    combined_train_x.append(train_data_per_domain[domain])
    combined_val_x.append(val_data_per_domain[domain])
    combined_test_x.append(test_data_per_domain[domain])

print('Finished preprocessing the data for the combined network!')
print('Starting to assemble and optimize the meta-classifier...')

expert_net = Sequential()
expert_net.add(Merge(trained_models, mode='concat'))
expert_net.add(Dense(256))
expert_net.add(Activation('relu'))
expert_net.add(Dense(3, activation='softmax'))
expert_net = Model.compile(expert_net)

class_weights = compute_class_weights(y_train_true)
validation_data = (combined_val_x, to_categorical(y_val_true, 3))
early_stopping = EarlyStopping(patience=50, verbose=1, mode='max', monitor='val_f1_score_pos_neg')
model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_path, mode='max', save_best_only=True,
                                   monitor='val_f1_score_pos_neg')

with open(model_json_path, 'w+') as f:
    f.write(expert_net.to_json())

y_train_true = to_categorical(y_train_true, 3)

history = expert_net.fit(combined_train_x, y_train_true,
                         validation_data=validation_data,
                         nb_epoch=1, batch_size=500,
                         callbacks=[early_stopping, model_checkpoint])

expert_net.load_weights(model_checkpoint_path)
score = expert_net.evaluate(combined_test_x, y_train_true)

metrics = {}
metrics_names = expert_net.metrics_names

for i in range(len(metrics_names)):
    name = metrics_names[i]
    metrics[name] = score[i]

metrics['history'] = history.history

with open(metrics_json_path, 'w+') as f:
    f.write(json.dumps(metrics))

with open(config_json_path, 'w+') as f:
    f.write(json.dumps({
        'test_data_path': test_data_path,
        'valid_data_path': val_data_path,
        'train_data_path': train_data_path
    }))

print('Finished optimizing the meta-classifier! Exiting...')
exit(0)
