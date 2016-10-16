import _pickle as pickle
import numpy as np
import os
import json

from data_loader import DataLoader
from model import Model

from os import path

from sklearn.cross_validation import StratifiedKFold

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical

class Executor(object):
    '''This class is responsible for loading all necessary data
       and training / validating a given model.'''

    #
    # Constants
    #
    RESULTS_DIRECTORY = path.realpath(
        path.join(path.dirname(__file__), '../results'))

    DEFAULT_PARAMS = {
        'batch_size': 500,
        'nb_epoch': 100,
        'nb_kfold_cv': 1,
        'validation_split': 0.0,
        'monitor_metric': 'val_f1_score_pos_neg',
        'model_checkpoint_monitor_metric': 'val_f1_score_pos_neg',
        'monitor_metric_mode': 'max'
    }

    def __init__(self, name, params):
        '''Constructor for the TestRun class.'''
        self.params = params
        self.name = name

        # merge with default params
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(params)

        self.nb_epoch = int(self.params['nb_epoch'])
        self.batch_size = int(self.params['batch_size'])
        self.nb_kfold_cv = int(self.params['nb_kfold_cv'])
        self.validation_split = float(self.params['validation_split'])
        self.monitor_metric = self.params['monitor_metric']
        self.monitor_metric_mode = self.params['monitor_metric_mode']
        self.model_checkpoint_monitor_metric = self.params['model_checkpoint_monitor_metric']

        self.results_path = path.join(self.RESULTS_DIRECTORY, name)
        self.weights_path = path.join(self.results_path, 'weights_%s.h5')
        self.model_path = path.join(self.results_path, 'model.json')
        self.params_path = path.join(self.results_path, 'params.json')
        self.validation_metrics_path = path.join(self.results_path,
                                                 'validation_metrics.json')

        self.train_metrics_path = path.join(self.results_path,
                                            'train_metrics_all.json')

        self.train_metrics_opt_path = path.join(self.results_path,
                                                'train_metrics_opt.json')

        self.create_results_directories()

    def run(self):
        '''Starts the experiment with the given params.'''
        self.log('Starting run...')
        self.store_params()

        test_data = self.params['test_data']
        validation_data_path = self.params['validation_data_path']
        vocabulary_path = self.params['vocabulary_path']
        vocab_emb_path = self.params['vocabulary_embeddings']
        vocab_emb = np.load(vocab_emb_path)

        self.log('Loading test data')

        vocabulary = self.load_vocabulary(vocabulary_path)
        sentiments, texts, nlabels = DataLoader.load(test_data, vocabulary)

        self.log('Test data loaded')

        if self.nb_kfold_cv > 1:
            self.log('Using %d-fold cross-validation' % self.nb_kfold_cv)

        count = 1
        histories = {}
        scores = []
        model_stored = False
        data_iter = None

        if self.nb_kfold_cv > 1:
            data_iter = StratifiedKFold(sentiments, n_folds=self.nb_kfold_cv)
        else:
            data_iter = [[range(0, len(sentiments)), []]]

        for train, test in data_iter:
            self.log('Loading model (round #%d)' % count)

            curr_model = Model(self.name, vocab_emb).build()

            # store the model only on the first iteration
            if not model_stored:
                self.store_model(curr_model)
                model_stored = True

            self.log('Model loaded (round #%d)' % count)

            X_train = texts[train]
            X_test = []

            if len(test) > 0:
                X_test = texts[test]

            Y_train = sentiments[train]
            Y_test = []

            if len(test) > 0:
                Y_test = sentiments[test]

            self.log('Start training (round #%d)' % count)

            history = self.train(curr_model, X_train, Y_train, X_test, Y_test, count)
            histories[count] = history.history

            self.log('Finished training (round #%d)' % count)
            self.log('Validating trained model (round #%d)' % count)

            curr_model.load_weights(self.weights_path % count)

            if len(X_test) > 0 and len(Y_test) > 0:
                score = curr_model.evaluate(X_test, to_categorical(Y_test), verbose=1)
                scores.append(score)

                self.log('Finished validating trained model (round #%d)' % count)

            count += 1

        self.store_validation_results(curr_model, scores)

        # Now we check all histories and only keep the model with the best f1 score
        monitor_metric_opt = 0.0
        monitor_metric_opt_nr = -1

        for nr, metrics in histories.items():
            if self.monitor_metric not in metrics:
                raise Exception('the metric "%s" is not available!' % self.monitor_metric)

            values = metrics[self.monitor_metric]
            store = False

            for v in values:
                store = self.monitor_metric_mode == 'max' and v > monitor_metric_opt
                store = store or self.monitor_metric_mode == 'min' and v < monitor_metric_opt

                if store:
                    monitor_metric_opt = v
                    monitor_metric_opt_nr = nr

        self.log('Looks like run #%d was the most successful with %s=%f' %
                 (monitor_metric_opt_nr, self.monitor_metric, monitor_metric_opt))

        self.store_histories(histories, monitor_metric_opt_nr)


    def train(self, m, X_train, Y_train, X_test, Y_test, count):
        '''This method trains the given model.'''
        validation_data = ()
        validation_split = None

        if len(X_test) > 0 and len(Y_test) > 0:
            validation_data = (X_test, to_categorical(Y_test))
        elif self.validation_split > 0.0:
            validation_split = self.validation_split

        return m.fit(X_train, to_categorical(Y_train),
                     validation_data=validation_data,
                     nb_epoch=self.nb_epoch,
                     validation_split=validation_split,
                     batch_size=self.batch_size,
                     callbacks=self.get_callbacks(count))

    def get_callbacks(self, counter):
        '''Creates the necessary callbacks for the keras model
           and returns them.'''
        return [self.create_early_stopping(), self.create_model_checkpoint(counter)]

    def create_model_checkpoint(self, count):
        '''Creates the model checkpoint callback for the model.'''
        return ModelCheckpoint(filepath=self.weights_path % count,
                               mode='max', save_best_only=True,
                               monitor=self.model_checkpoint_monitor_metric)

    def create_early_stopping(self):
        '''Creates the early stopping callback for the model.'''
        return EarlyStopping(monitor='val_f1_score', patience=50,
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

    def store_histories(self, histories, opt_nr):
        '''Stores the history of learning as a npy file at the train_metrics_path.'''
        with open(self.train_metrics_opt_path, 'w+') as f:
            f.write(json.dumps(histories[opt_nr]))

        with open(self.train_metrics_path, 'w+') as f:
            f.write(json.dumps(histories))

    def store_validation_results(self, model, scores):
        '''Stores the average over multiple evaluation scores as a JSON file.'''
        avg_metrics = {}
        all_metrics = []
        metrics_names = model.metrics_names

        for i in range(0, len(scores)):
            all_metrics.append({})

        for i in range(0, len(metrics_names)):
            name = metrics_names[i]

            for j, s in enumerate(scores):
                if name not in avg_metrics:
                    avg_metrics[name] = []

                avg_metrics[name].append(s[i])
                all_metrics[j][name] = s[i]
                all_metrics[j]['round'] = j + 1

        for n in metrics_names:
            avg_metrics['%s_std' % n] = np.std(avg_metrics[n])
            avg_metrics['%s_mean' % n] = np.mean(avg_metrics[n])
            avg_metrics[n] = np.sum(avg_metrics[n]) / len(scores)

        final_metrics = {
            'avg': avg_metrics,
            'all': all_metrics
        }

        with open(self.validation_metrics_path, 'w+') as f:
            f.write(json.dumps(final_metrics))

    def create_results_directories(self):
        '''This function is responsible for creating the results directory.'''
        if not path.isdir(self.results_path):
            os.mkdir(self.results_path)

    def log(self, msg):
        '''Simple logging function which only works in verbose mode.'''
        print('[%s] %s' % (self.name, msg))
