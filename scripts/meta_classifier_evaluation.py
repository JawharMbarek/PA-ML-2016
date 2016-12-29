#!/usr/bin/env python

import sys
import os
import tempfile

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
import gc
import h5py

from keras.models import Sequential
from keras.layers import Merge, Dense, Activation, Flatten, MaxPooling1D, Convolution1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from data_utils import compute_class_weights, powerset
from data_loader import DataLoader
from evaluation_metrics import f1_score_pos_neg
from model import Model

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Missing mandatory argument')
    print('       python scripts/meta_classifier_evaluation.py <opt-tsv> <val-tsv> <test-tsv>')
    sys.exit(2)

np_seed = int(time.time())

if 'NP_RAND_SEED' in os.environ:
    np_seed = int(os.environ['NP_RAND_SEED'])
    print('Using injected seed %d' % np_seed)
else:
    print('Using the unix timestamp %d as a seed' % np_seed)

np.random.seed(np_seed)

ALL_DOMAINS = ['dai', 'dil', 'hul', 'mpq', 'semeval', 'tac']
SENTENCE_LENGTH = 140
ROOT_PATH = path.abspath(path.join(path.dirname(__file__), '..'))
MODELS_PATH = path.join(ROOT_PATH, 'models')
VOCABS_PATH = path.join(ROOT_PATH, 'vocabularies')
META_RESULTS_PATH = path.join(ROOT_PATH, 'meta-results')

if not path.isdir(META_RESULTS_PATH):
    os.mkdir(META_RESULTS_PATH)

train_data_path = argv[0]
val_data_path = argv[1]
test_data_path = argv[2]
used_domains = argv[3].split(',')

def get_domain_model_dir(d):
    return path.join(MODELS_PATH, 'best_model_crossdomain_we_ds_%s' % d)

min_included_domains = 4

# domain_combinations = powerset(ALL_DOMAINS)
# domain_combinations = filter(lambda x: len(x) >= min_included_domains, domain_combinations)
# domain_combinations = list(domain_combinations)

vocabs  = ['vocab_en300M_reduced.pickle', 'vocab_news_emb.pickle', 'vocab_wiki_emb.pickle']

print('Loading models...')

models = {}

nb_of_layers_to_remove = 6

for i, domain in enumerate(ALL_DOMAINS):
    model_dir = get_domain_model_dir(domain)
    model_json_file = path.join(model_dir, 'model.json')
    model_weights_file = path.join(model_dir, 'weights_1.h5')

    with open(model_json_file, 'r') as f:
        models[domain] = keras.models.model_from_json(f.read())
    
    models[domain].load_weights(model_weights_file, by_name=True)

    # rename layers to mitigate the unique name problem
    # see https://github.com/fchollet/keras/issues/3974
    for j in range(len(models[domain].layers)):
        models[domain].layers[j].name += '_%s_model' % domain

    # Remove classification part of each model
    for i in range(nb_of_layers_to_remove):
        models[domain].layers.pop()

    if not models[domain].layers:
        models[domain].outputs = []
        models[domain].inbound_nodes = []
        models[domain].outbound_nodes = []
    else:
        models[domain].layers[-1].outbound_nodes = []
        models[domain].outputs = [models[domain].layers[-1].output]

    # Nasty stuff to ensure that the model is loaded properly
    # after removing the last three layers needed for classification...
    with tempfile.NamedTemporaryFile() as temp_config:
        with tempfile.NamedTemporaryFile() as temp_weights:
            temp_config.write(models[domain].to_json().encode())
            models[domain].save_weights(temp_weights.name)

            used_layer_names = list(map(lambda x: x.name, models[domain].layers))
            
            with h5py.File(temp_weights.name) as temp_weights_file:
                # Delete the weights of all layers which won't be needed in the
                # shrinked model because keras complains otherwise
                for group_key in temp_weights_file.keys():
                    if group_key not in used_layer_names:
                        del temp_weights_file[group_key]

                # MORE nasty stuff I don't want to talk about...
                temp_weights_file.attrs['layer_names'] = [x for x in temp_weights_file.attrs['layer_names'] if x.decode() in used_layer_names]

                temp_config.seek(0)
                models[domain] = keras.models.model_from_json(temp_config.read().decode())
                models[domain].load_weights(temp_weights.name)

print('Models loaded!')

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

        test_data_per_vocab[vocab_length] = test_txts
        val_data_per_vocab[vocab_length] = val_txts
        train_data_per_vocab[vocab_length] = train_txts

        if y_train_true is None:
            y_train_true = train_sents
            y_val_true = val_sents
            y_test_true = test_sents

class_weights = compute_class_weights(y_train_true)

y_train_true = to_categorical(y_train_true, 3)
y_test_true = to_categorical(y_test_true, 3)
y_val_true = to_categorical(y_val_true, 3)

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

timestamp = int(time.time())
current_results_path = path.join(
    META_RESULTS_PATH, 'run_target_%s_used_%s_%d' % (
        target_domain, '-'.join(list(used_domains)), timestamp
    )
)
model_checkpoint_path = path.join(current_results_path, 'expert_net_weights_%s_%d.h5' % (target_domain, timestamp))
model_json_path = path.join(current_results_path, 'experts_net_models_%s_%d.json' % (target_domain, timestamp))
metrics_json_path = path.join(current_results_path, 'experts_net_metrics_%s_%d.json' % (target_domain, timestamp))
config_json_path = path.join(current_results_path, 'experts_net_config_%s_%d.json' % (target_domain, timestamp))

if not path.isdir(current_results_path):
    os.mkdir(current_results_path)

print('Starting to assemble and optimize the meta-classifier...')

for k in models.keys():
    models[k] = Model.compile(models[k])

expert_net = Sequential()
expert_net.add(Merge(trained_models, mode='concat'))
expert_net.add(Flatten())
expert_net.add(Dense(200))
expert_net.add(Dropout(0.2))
expert_net.add(Activation('relu'))
expert_net.add(Dense(3, activation='softmax'))
expert_net = Model.compile(expert_net)

validation_data = (combined_val_x, y_val_true)
early_stopping = EarlyStopping(patience=50, verbose=1, mode='max', monitor='val_f1_score_pos_neg')
model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_path, mode='max', save_best_only=True,
                                   monitor='val_f1_score_pos_neg')

with open(model_json_path, 'w+') as f:
    f.write(expert_net.to_json())

history = expert_net.fit(combined_train_x, y_train_true,
                         validation_data=validation_data,
                         nb_epoch=1000, batch_size=300,
                         class_weight=class_weights,
                         callbacks=[early_stopping, model_checkpoint])

expert_net.load_weights(model_checkpoint_path)
score = expert_net.evaluate(combined_test_x, y_test_true)

metrics = {}
metrics_names = expert_net.metrics_names

for i in range(len(metrics_names)):
    name = metrics_names[i]
    metrics[name] = score[i]

metrics['history'] = history.history

with open(metrics_json_path, 'w+') as f:
    f.write(json.dumps(metrics, sort_keys=True, indent=4))

with open(config_json_path, 'w+') as f:
    f.write(json.dumps({
        'used_domain_models': used_domains,
        'test_data_path': test_data_path,
        'valid_data_path': val_data_path,
        'train_data_path': train_data_path,
        'np_seed': np_seed
    }, sort_keys=True, indent=4))

print('Finished optimizing the meta-classifier! Exiting...')
