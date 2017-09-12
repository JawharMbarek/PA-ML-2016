import sys
import os
import keras
import pickle
import csv

from os import path
from keras.utils.np_utils import to_categorical

# Hack to be able to import DataLoader / parse_utils
dlp = path.realpath(path.join(path.dirname(__file__), '..', 'source'))
sys.path.insert(0, dlp)

from data_utils import compute_class_weights

# DAI -> Twitter distant, Twitter embeddings
# DIL -> Reviews distant, Twitter embeddings
# HUL -> Reviews distant, Wiki embeddings
# JCR -> Reviews distant, Wiki embeddings
# MPQ -> None distant, News embeddings
# SEv -> Twitter distant, News embeddings
# SEM -> Twitter distant, Twitter embeddings
# TAC -> Reviews distant, Random embeddings

# Hack to be able to import DataLoader / parse_utils
dlp = path.realpath(path.join(path.dirname(__file__), '..', 'source'))
sys.path.insert(0, dlp)

from data_loader import DataLoader

argv = sys.argv[1:]

if len(argv) < 4:
    print('ERROR: Missing mandatory arguments!')
    print('       (./generate_predictions_with_model.py <model-dir> <voc-path> <in-csv-path> <out-csv-path>')
    sys.exit(2)

model_dir_path = argv[0]
vocab_path = argv[1]
in_data_path = argv[2]
out_pred_path = argv[3]

model = None
vocab = None

model_conf_path = path.join(model_dir_path, 'model.json')
model_weights_path = path.join(model_dir_path, 'weights_1.h5')

print('Loading model and vocabulary...')

with open(model_conf_path, 'r') as conf_f:
    model = keras.models.model_from_json(conf_f.read())

with open(vocab_path, 'rb') as voc_f:
    vocab = pickle.load(voc_f)

print('Model and vocabulary loaded!')
print('Loading data...')

data_sents, data_txts, data_raw_txts, _ = DataLoader.load(in_data_path, vocab, randomize=True)
data_sents = to_categorical(data_sents, 3)

print('Data loaded!')
print('Starting predictions...')

result = model.predict(data_txts)

with open(out_pred_path, 'w+') as out_f:
    csv_writer = csv.writer(out_f)
    csv_writer.writerow(['Text', 'Sentiment', 'Prediction'])

    for txt, sent, pred in zip(data_raw_txts, data_sents, result):
        csv_writer.writerow([txt[2], sent, pred])

print('Predictions generated and stored in %s!' % out_pred_path)