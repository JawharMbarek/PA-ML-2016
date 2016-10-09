from keras.models import Sequential

from keras.layers import Dense, Activation, ZeroPadding1D, Dropout
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, Flatten

from keras.optimizers import Adadelta

from data_utils import fbeta_score


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

    def __init__(self, name, vocab_emb, verbose=True):
        '''Constructor for the model class.'''
        self.vocabulary_embeddings = vocab_emb
        self.max_features = vocab_emb.shape[0]
        self.verbose = verbose

    def build(self):
        '''This function builds the actual keras model which is then used
           by train() and validate() to run it.'''

        embeddings = Embedding(
            self.max_features,
            self.nb_embedding_dims,
            input_length=self.input_maxlen,
            weights=[self.vocabulary_embeddings],
            dropout=0.2,
        )

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

        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=adadelta, metrics=['accuracy', fbeta_score])

        return model
