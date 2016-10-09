from __future__ import print_function
import numpy as np
import os
import getopt
import sys
from data_utils import tsv_sentiment_loader
import matplotlib.pyplot as plt
from nltk import TweetTokenizer
import _pickle as cPickle

from keras.models import Sequential
from keras.layers import Dense, Activation, ZeroPadding1D, Dropout
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical



embedding_fname = ''
language = ''
vocab_fname = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "e:l:v:", ["embedding=", "language=", "vocab="])
except getopt.GetoptError:
    print('Bad Input Args')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-e", "--embedding"):
        embedding_fname = '{}.npy'.format(arg)
    elif opt in ("-v", "--vocab"):
        vocab_fname = '{}.pickle'.format(arg)
    elif opt in ("-l", "--language"):
        language = arg

vocab_path = os.path.join('vocabularies', vocab_fname)

model_dir = './model/{}'.format(language)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

res_dir = 'results/{}'.format(language)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

super_model_path = './model/{}/super_phase_model.json'.format(language)
super_weight_path = './model/{}/super_phase_weights.h5'.format(language)

#
# hyperparameters
#
input_maxlen = 140
nb_filter = 200
nb_batch_size = 500
nb_embedding_dims = 52
filter_length = 6
hidden_dims = nb_filter
nb_epoch = 100

print('Loading Embeddings...')

fname_wordembeddings = os.path.join('embeddings', embedding_fname)
print(fname_wordembeddings)
vocab_emb = np.load(fname_wordembeddings)

print("Word embedding matrix size:", vocab_emb.shape)
max_features = vocab_emb.shape[0]

tknzr = TweetTokenizer(reduce_len=True)
alphabet = cPickle.load(open(vocab_path, 'rb'))
dummy_word_idx = alphabet.get('DUMMY_WORD_IDX', 1)
print("alphabet", len(alphabet))
print('dummy word: ', dummy_word_idx)

tids, sentiments, text, nlabels = tsv_sentiment_loader('semeval/crossdomain/DAI_tweets_full.tsv', alphabet, tknzr)

##Layers
print('Build Model...')

embeddings = Embedding(
    max_features,
    nb_embedding_dims,
    input_length=input_maxlen,
    weights=[vocab_emb],
    dropout=0.2,
)


zeropadding = ZeroPadding1D(filter_length - 1)


conv1 = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1
)

conv2 = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1
)

max_pooling1 = MaxPooling1D(pool_length=4,stride=2)

model = Sequential()
model.add(embeddings)
model.add(zeropadding)
model.add(conv1)
model.add(max_pooling1)
model.add(conv2)

# we need to know the output length of the last convolution layer
max_pooling2 = MaxPooling1D(pool_length=model.layers[-1].output_shape[1])
model.add(max_pooling2)

model.add(Flatten())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(nlabels, activation='softmax'))

model.summary()
adadelta = Adadelta(lr=1.0,rho=0.95,epsilon=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=adadelta, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='max')
model_checkpoit = ModelCheckpoint(filepath=super_weight_path,verbose=1, save_best_only=True, monitor='val_acc', mode='max')
history = model.fit(text, to_categorical(sentiments),
                    nb_epoch=nb_epoch, batch_size=500,
                    verbose=1, validation_split=0.2,
                    callbacks=[early_stop, model_checkpoit])

print('Storing Model')
json_string = model.to_json()
open(super_model_path, 'w+').write(json_string)

print('Load Model')
model.load_weights(super_weight_path)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()