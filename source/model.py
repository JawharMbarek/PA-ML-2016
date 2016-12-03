from keras.models import Sequential

from keras.layers import Dense, Activation, ZeroPadding1D, Dropout
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, Flatten

from keras.optimizers import Adadelta

import evaluation_metrics as ev

class Model(object):
    '''This class is responsible for storing our current the keras model.
       Via the function build(), the keras representation of this model
       can be created.'''

    # hyperparameters
    input_maxlen = 140
    nb_filter = 200
    nb_embedding_dims = 52
    filter_length = 6
    hidden_dims = nb_filter
    nb_labels = 3

    def __init__(self, name, vocab_emb, input_maxlen=140,
                 random_embeddings=False, verbose=False):
        '''Constructor for the model class.'''
        self.vocabulary_embeddings = vocab_emb
        self.max_features = vocab_emb.shape[0]
        self.random_embeddings = random_embeddings
        self.verbose = verbose
        self.input_maxlen = input_maxlen

    def build(self, id=1):
        '''This function builds the actual keras model which is then used
           by train() and validate() to run it.'''
        build_meth = 'build%d' % id

        if build_meth in dir(self):
            return getattr(self, build_meth)()
        else:
            raise Exception('model version %s is not possible' % str(id))

    def build1(self):
        '''Builds the version 1 of the model.'''
        embeddings = self._get_embeddings_layer()

        zeropadding = ZeroPadding1D(self.filter_length - 1)

        conv1 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        conv2 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        max_pooling1 = MaxPooling1D(pool_length=4, stride=2)

        model = Sequential()
        model.add(embeddings)
        model.add(zeropadding)
        model.add(conv1)
        model.add(max_pooling1)
        model.add(conv2)

        # we need to know the output length of the last convolution layer
        prev_pool_length = model.layers[-1].output_shape[1]
        max_pooling2 = MaxPooling1D(pool_length=prev_pool_length)
        model.add(max_pooling2)

        model.add(Flatten())
        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_labels, activation='softmax'))

        if self.verbose:
            model.summary()

        return Model.compile(model)

    def build2(self):
        '''Returns the version 2 of the model. This one uses
           three conv layers instead of just two.'''
        embeddings = self._get_embeddings_layer()

        zeropadding = ZeroPadding1D(self.filter_length - 1)

        conv1 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        max_pooling1 = MaxPooling1D(pool_length=4, stride=2)

        conv2 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        max_pooling2 = MaxPooling1D(pool_length=4, stride=2)

        conv3 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        model = Sequential()
        model.add(embeddings)
        model.add(zeropadding)
        model.add(conv1)
        model.add(max_pooling1)
        model.add(conv2)
        model.add(max_pooling2)
        model.add(conv3)
        # we need to know the output length of the last convolution layer
        prev_pool_length = model.layers[-1].output_shape[1]
        max_pooling3 = MaxPooling1D(pool_length=prev_pool_length)
        model.add(max_pooling3)
        model.add(Flatten())
        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_labels, activation='softmax'))

        if self.verbose:
            model.summary()

        return Model.compile(model)

    def build3(self):
        '''Returns the version 3 of the model. This one uses
           four conv layers instead of just two.'''
        embeddings = self._get_embeddings_layer()

        zeropadding = ZeroPadding1D(self.filter_length - 1)

        conv1 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        max_pooling1 = MaxPooling1D(pool_length=4, stride=2)

        conv2 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        max_pooling2 = MaxPooling1D(pool_length=4, stride=2)

        conv3 = Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        max_pooling3 = MaxPooling1D(pool_length=4, stride=2)

        conv4= Convolution1D(
            nb_filter=self.nb_filter,
            filter_length=self.filter_length,
            border_mode='valid',
            activation='relu',
            subsample_length=1
        )

        model = Sequential()
        model.add(embeddings)
        model.add(zeropadding)
        model.add(conv1)
        model.add(max_pooling1)
        model.add(conv2)
        model.add(max_pooling2)
        model.add(conv3)
        model.add(max_pooling3)
        model.add(conv4)
        # we need to know the output length of the last convolution layer
        prev_pool_length = model.layers[-1].output_shape[1]
        max_pooling4 = MaxPooling1D(pool_length=prev_pool_length)
        model.add(max_pooling4)
        model.add(Flatten())
        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_labels, activation='softmax'))

        if self.verbose:
            model.summary()

        return Model.compile(model)

    @staticmethod
    def compile(model):
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=adadelta, metrics=['accuracy',
                      ev.f1_score, ev.f1_score_pos, ev.f1_score_neg,
                      ev.f1_score_pos_neg, ev.f1_score_neu])

        return model

    def _get_embeddings_layer(self):
        '''Returns the embedding layer for the model. It returns
           the embedding layer initialized with the given word
           embeddings, otherwise it is randomly and uniformly initialized.'''
        weights = None

        if not self.random_embeddings:
            weights = [self.vocabulary_embeddings]
        else:
            print('INFO: Using random word embeddings')

        return Embedding(self.max_features, self.nb_embedding_dims,
                         input_length=self.input_maxlen, dropout=0.2,
                         weights=weights)