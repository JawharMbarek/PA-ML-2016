from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical

from nltk import TweetTokenizer
from data_utils import tsv_sentiment_loader, fbeta_score

import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy
import os
import sys
import getopt

model_path = ''
weights_path = ''
data_path = ''
vocab_path = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "m:w:d:v:", ["model=", "weights=", 'data=', 'vocab='])
except getopt.GetoptError:
    print('Bad Input Args')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-m", "--model"):
        model_path = arg
    elif opt in ("-w", "--weights"):
        weights_path = arg
    elif opt in ('-d', '--data'):
        data_path = arg
    elif opt in ('-v', '--vocab'):
        vocab_path = arg

# load json and create model
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model
print(weights_path)
loaded_model.load_weights(weights_path)
print("Loaded model from disk")

tknzr = TweetTokenizer(reduce_len=True)

# load datase
alphabet = cPickle.load(open(vocab_path, 'rb'))
tids, sentiments, texts, nlabels = tsv_sentiment_loader(data_path, alphabet, tknzr)
dummy_word_idx = alphabet.get('DUMMY_WORD_IDX', 1)

# evaluate loaded model on test data
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=adadelta, metrics=['accuracy', fbeta_score])

score = loaded_model.evaluate(texts, to_categorical(sentiments), verbose=1)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
print("%s: %.2f%%" % (loaded_model.metrics_names[2], score[2] * 100))
