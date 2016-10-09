import _pickle as pickle
import numpy as np
import os
import json

from model import Model
from os import path

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical

from data_utils import tsv_sentiment_loader

from nltk import TweetTokenizer


class Executor(object):
    '''This class is responsible for loading all necessary data
       and training / validating a given model.'''

    #
    # Constants
    #
    RESULTS_DIRECTORY = path.realpath(
        path.join(path.dirname(__file__), '../results'))

    #
    # default params
    #
    batch_size = 500
    nb_epoch = 100
    validation_split = 0.2

    def __init__(self, name, params):
        '''Constructor for the TestRun class.'''
        self.params = params
        self.name = name

        self.results_path = path.join(self.RESULTS_DIRECTORY, name)
        self.weights_path = path.join(self.results_path, 'weights.h5')
        self.model_path = path.join(self.results_path, 'model.json')
        self.params_path = path.join(self.results_path, 'params.json')

        self.validation_metric_path = path.join(self.results_path,
                                                'validation_metrics.json')

        self.history_path = path.join(self.results_path,
                                      'test_metrics.json')

        self.create_results_directories()

    def run(self):
        '''Starts the experiment with the given params.'''
        self.log('Starting run...')

        self.log('Loading current model')

        test_data_path = self.params['test_data_path']
        validation_data_path = self.params['validation_data_path']
        vocabulary_path = self.params['vocabulary_path']
        vocab_emb_path = self.params['vocabulary_embeddings']

        curr_model = Model(self.name, np.load(vocab_emb_path)).build()

        self.log('Model loaded')

        self.log('Loading test data')

        vocabulary = self.load_vocabulary(vocabulary_path)

        tids, sentiments, texts, nlabels = self.load_test_data(
            test_data_path, vocabulary
        )

        self.log('Test data loaded')

        self.log('Start training')
        self.train(curr_model, texts, sentiments)
        self.log('Finished training')

        self.store_params()
        self.store_model(curr_model)

        if 'validation_data_path' in self.params:
            self.log('Starting validation on %s' % validation_data_path)

            tids, sentiments, texts, nlabels = self.load_test_data(
                validation_data_path, vocabulary
            )

            self.validate(curr_model, texts, sentiments)

            self.log('Finished validation')

    def train(self, m, X, Y):
        '''This method trains the given model.'''
        history = m.fit(X, to_categorical(Y),
                        nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                        verbose=1, validation_split=self.validation_split,
                        callbacks=self.get_callbacks())

        self.store_history(history)

    def validate(self, m, X_val, Y_val):
        '''This method validates a trained model.'''
        score = m.evaluate(X_val, to_categorical(Y_val), verbose=1)

        # newline required since keras omits it..
        print('')

        for i in range(0, len(m.metrics_names)):
            print("%s: %.2f" % (m.metrics_names[i], score[i]))

    def get_callbacks(self):
        '''Creates the necessary callbacks for the keras model
           and returns them.'''
        return [self.create_early_stopping(), self.create_model_checkpoint()]

    def create_model_checkpoint(self):
        '''Creates the model checkpoint callback for the model.'''
        return ModelCheckpoint(filepath=self.weights_path, verbose=1,
                               save_best_only=True, monitor='val_acc',
                               mode='max')

    def create_early_stopping(self):
        '''Creates the early stopping callback for the model.'''
        return EarlyStopping(monitor='val_acc', patience=50,
                             verbose=1, mode='max')

    def load_test_data(self, path, vocabulary):
        '''Loads the file at the given path with the tsv loader.'''
        tokenizer = TweetTokenizer(reduce_len=True)
        return tsv_sentiment_loader(path, vocabulary, tokenizer)

    def load_vocabulary(self, path):
        '''Loads the vocabulary stored as a pickle file.'''
        return pickle.load(open(path, 'rb'))

    def store_model(self, m):
        '''Stores the given model as a json file a the model_path.'''
        with open(self.model_path, 'w+') as f:
            f.write(m.to_json())

    def store_params(self):
        '''Stores the current params as a json file at the params_path.'''
        with open(self.params_path, 'w+') as f:
            f.write(json.dumps(self.params))

    def store_history(self, history):
        '''Stores the history of learning as a npy file at the history_path.'''
        with open(self.history_path, 'w+') as f:
            f.write(json.dumps(history.history))

    def create_results_directories(self):
        '''This function is responsible for creating the results directory.'''
        if not path.isdir(self.results_path):
            os.mkdir(self.results_path)

    def log(self, msg):
        '''Simple logging function which only works in verbose mode.'''
        print('[%s] %s' % (self.name, msg))
