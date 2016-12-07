import pickle
import numpy as np
import os
import json
import itertools
import time
import h5py
import keras

from data_utils import compute_class_weights
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
        # Batch size used while training the model
        'batch_size': 500,

        # Number of epochs to train the model at max
        'nb_epoch': 100,

        # Number folds to use when using k-fold cross validation
        'nb_kfold_cv': 1,

        # Percentage of the test data which should be used as validation
        # data. This configuration is ignored inn case 'validation_data_path'
        # is set.
        'validation_split': 0.0,

        # Metric to choose the best fold from when using k-fold cv
        'monitor_metric': 'val_f1_score_pos_neg',

        # Can be 'max' or 'min' and decides wether the maximum
        # or minimum value is considered 'best' when monitoring
        # the metrics for each fold.
        'monitor_metric_mode': 'max',

        # Metric to choose the best model from. It stores the best
        # model as a json with the associated weights as h5 file
        # in the results directory.
        'model_checkpoint_monitor_metric': 'val_f1_score_pos_neg',

        # Metric which is used in the EarlyStopping callback. It
        # monitors the given metric and stops the training after
        # 50 consecutive epochs without an improvement.
        'early_stopping_monitor_metric': 'val_f1_score_pos_neg',

        # Set the patience parameter in the EarlyStopping callback.
        # Basically, the model is trained until there's been no
        # improvement in the early_stopping_monitor_metric for
        # early_stopping_patience consecutive epochs.
        'early_stopping_patience': 75,

        # Decides wether the loaded train data is shuffled before
        # it is returned by the Dataloader.
        'randomize_test_data': True,

        # Decides wether class weights should be set while training
        # to compensate if train data with skewed classes is used.
        'set_class_weights': False,

        # Decides which model is loaded by the Executor. See the
        # function build() in the Model class for more details.
        'model_id': 1,

        # The training can also be done by using and HDF5 file as
        # the training data. In this case there's no reason for the
        # preprocessing and hence speeds up the training process. The
        # HDF5 should provide the two datasets 'x' and 'y' where the
        # first contains the input values and the latter the solutions.
        'use_preprocessed_data': False,

        # Path to the HDF5 file to use as preprocessed data.        
        'preprocessed_data': None,

        # If preprocessed data is used, we need to set the samples per
        # epoch via the configuration to the number of training examples.
        # This is the case because a generator is used and keras needs to
        # know the number of examples it should pull from the iterator.
        'samples_per_epoch': 100000,

        # Data to use for validation.
        'validation_data_path': None,

        # The maximum length of the input 'sentences'.
        'max_sent_length': 140,

        # If set, the model will be loaded from this JSON file instead
        # of the build() function in the Model class.
        'model_json_path': None,

        # If set, the model will be initialized with the weights stored
        # in the HDF5 file located at the given path.
        'model_weights_path': None,

        # If set to false, no validation will be done while training but
        # instead only at the end after the training has finished.
        'validate_while_training': True,

        # If set to true, the embeddings layer will be initialized with
        # random embeddings instead of using the precalculated word2vec
        # embeddings
        'use_random_embeddings': False
    }

    def __init__(self, name, params):
        '''Constructor for the TestRun class.'''
        self.params = params
        self.name = name

        # merge with default params
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(params)

        if 'group_id' in params:
            self.group_id = params['group_id']
        else:
            self.group_id = ''

        self.nb_epoch = int(self.params['nb_epoch'])
        self.batch_size = int(self.params['batch_size'])
        self.nb_kfold_cv = int(self.params['nb_kfold_cv'])
        self.validation_split = float(self.params['validation_split'])
        self.randomize_test_data = self.params['randomize_test_data']
        self.set_class_weights = self.params['set_class_weights']
        self.model_id = self.params['model_id']
        self.max_sent_length = self.params['max_sent_length']
        self.validate_while_training = self.params['validate_while_training']

        self.monitor_metric = self.params['monitor_metric']
        self.monitor_metric_mode = self.params['monitor_metric_mode']
        self.model_checkpoint_monitor_metric = self.params['model_checkpoint_monitor_metric']
        self.early_stopping_monitor_metric = self.params['early_stopping_monitor_metric']
        self.early_stopping_patience = self.params['early_stopping_patience']

        if not self.validate_while_training and self.validation_split == 0.0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('! WARNING: Validation while training is disabled, cannot us validation F1 scores as metrics. !')
            print('!          All "val_" in metrics names will be replaced before the training is started.      !')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            self.monitor_metric = self.monitor_metric.replace('val_', '')
            self.model_checkpoint_monitor_metric = self.model_checkpoint_monitor_metric.replace('val_', '')
            self.early_stopping_monitor_metric = self.early_stopping_monitor_metric.replace('val_', '')

        self.model_json_path = self.params['model_json_path']
        self.model_weights_path = self.params['model_weights_path']

        self.use_preprocessed_data = self.params['use_preprocessed_data']
        self.preprocessed_data = self.params['preprocessed_data']

        self.results_path = path.join(self.RESULTS_DIRECTORY, self.group_id, self.name)
        self.weights_path = path.join(self.results_path, 'weights_%s.h5')
        self.model_path = path.join(self.results_path, 'model.json')
        self.params_path = path.join(self.results_path, 'params.json')
        self.validation_metrics_path = path.join(self.results_path,
                                                 'validation_metrics.json')

        self.train_metrics_path = path.join(self.results_path,
                                            'train_metrics_all.json')

        self.train_metrics_opt_path = path.join(self.results_path,
                                                'train_metrics_opt.json')

        self.use_random_embeddings = self.params['use_random_embeddings']

        self.create_results_directories()

    def run(self):
        '''Starts the experiment with the given params.'''
        self.log('Starting run...')
        self.store_params()

        curr_model = None
        test_data = self.params['test_data']
        validation_data_path = self.params['validation_data_path']
        vocabulary_path = self.params['vocabulary_path']
        vocab_emb_path = self.params['vocabulary_embeddings']
        vocab_emb = np.load(vocab_emb_path)
        histories = {}
        scores = []
        monitor_metric_opt_nr = None

        vocabulary = self.load_vocabulary(vocabulary_path)

        if self.use_preprocessed_data:
            self.log('Loading model')

            curr_model = self.get_model(vocab_emb)

            self.store_model(curr_model)
            self.log('Using preprocessed data: %s' % self.preprocessed_data)

            with h5py.File(self.preprocessed_data) as f:
                train_x = f['x']
                train_y = f['y']

                final_weights_path = self.weights_path % ('%s_complete_final' % self.name)
                max_sent_length = len(train_x[0])

                while not path.isfile(final_weights_path):
                    def create_ndarray(max_size=max_sent_length):
                        return np.ndarray(shape=(self.batch_size, max_size), dtype=np.int)

                    def generator_function(bsize):
                        total_count = 0
                        base_idx = 0
                        start_time = time.time()

                        while total_count < len(train_x):
                            yield_x = create_ndarray()
                            yield_y = create_ndarray(3)

                            if bsize > (len(train_x) - total_count):
                                bsize = total_count - len(train_x)

                            for i in range(0, bsize):
                                yield_x[i] = train_x[base_idx + i]
                                yield_y[i] = train_y[base_idx + i]

                                total_count += 1

                            base_idx += bsize

                            yield yield_x, yield_y

                            if total_count % 20e6 == 0:
                                step = '%sM' % str(total_count)[0:2]
                                name = 'model_amazon_distant_complete_%sM' % step
                                curr_model.save_weights(self.weights_path % name)
                                self.log('Model saved after %s training examples' % step)

                    history = curr_model.fit_generator(
                        generator_function(self.batch_size),
                        len(train_x) / 10,
                        10, verbose=1, nb_worker=1
                    )

                    histories[1] = history.history
                    curr_model.save_weights(final_weights_path)

                self.log('Finished training')
        else:
            self.log('Loading test data')

            sents, txts, raw_data, nlabels = DataLoader.load(
                test_data, vocabulary,
                randomize=self.params['randomize_test_data']
            )

            sents_val, txts_val, raw_data_val, nlabels = DataLoader.load(
                validation_data_path, vocabulary,
                randomize=self.params['randomize_test_data']
            )

            self.store_tsv_data(raw_data, 'train')
            self.store_tsv_data(raw_data_val, 'validation')

            self.log('Test data loaded')

            if self.nb_kfold_cv > 1:
                self.log('Using %d-fold cross-validation' % self.nb_kfold_cv)

            count = 1
            scores = []
            model_stored = False
            data_iter = None
            if self.nb_kfold_cv > 1:
                data_iter = StratifiedKFold(sents, n_folds=self.nb_kfold_cv)
            else:
                data_iter = [[range(0, len(sents)), []]]

            for train, test in data_iter:
                self.log('Loading model (round #%d)' % count)

                curr_model = None

                if self.model_json_path and self.model_weights_path:
                    self.log('Using model at %s' % self.model_json_path)
                    self.log('Loading weights at %s' % self.model_weights_path)

                    with open(self.model_json_path, 'r') as f:
                        curr_model = keras.models.model_from_json(f.read())
                        curr_model.load_weights(self.model_weights_path)

                    curr_model = Model.compile(curr_model)
                else:
                    curr_model = self.get_model(vocab_emb)

                # store the model only on the first iteration
                if not model_stored:
                    self.store_model(curr_model)
                    model_stored = True

                self.log('Model loaded (round #%d)' % count)

                X_train = txts[train]
                X_test = txts_val

                Y_train = sents[train]
                Y_test = sents_val

                self.log('Start training (round #%d)' % count)

                if len(X_train) > 0:
                    history = self.train(curr_model, X_train, Y_train, X_test, Y_test, count)
                    histories[count] = history.history

                self.log('Finished training (round #%d)' % count)

                weights_path = self.weights_path % count

                if len(X_test) > 0 and len(Y_test) > 0:
                    self.log('Validating trained model (round #%d)' % count)

                    if path.isfile(weights_path):
                        curr_model.load_weights(self.weights_path % count)

                    score = curr_model.evaluate(X_test, to_categorical(Y_test), verbose=1)
                    scores.append(score)

                    self.log('Finished validating trained model (round #%d)' % count)

                count += 1

            if len(scores) > 0:
                self.store_validation_results(curr_model, scores)

            # Now we check all histories and only keep the model with the best f1 score
            monitor_metric_opt = -1.0
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
        class_weights = None
        result = None

        if self.set_class_weights:
            class_weights = compute_class_weights(Y_train)
        if len(X_test) > 0 and len(Y_test) > 0 and self.validate_while_training:
            validation_data = (X_test, to_categorical(Y_test))
        elif self.validation_split > 0.0:
            validation_split = self.validation_split

        return m.fit(X_train, to_categorical(Y_train, nb_classes=3),
                     validation_data=validation_data,
                     nb_epoch=self.nb_epoch,
                     validation_split=validation_split,
                     batch_size=self.batch_size,
                     callbacks=self.get_callbacks(count))

    def get_model(self, embeddings):
        return Model(self.name, embeddings, input_maxlen=self.max_sent_length,
                     random_embeddings=self.use_random_embeddings).build(self.model_id)

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
        return EarlyStopping(patience=self.early_stopping_patience, verbose=1,
                             mode='max', monitor=self.early_stopping_monitor_metric)

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
        if opt_nr is not None and opt_nr != -1:
            with open(self.train_metrics_opt_path, 'w+') as f:
                f.write(json.dumps(histories[opt_nr]))

        with open(self.train_metrics_path, 'w+') as f:
            f.write(json.dumps(histories))

    def store_validation_results(self, model, scores):
        '''Stores the average over multiple evaluation scores as a JSON file.'''
        avg_metrics = {}
        all_metrics = []
        metrics_names = model.metrics_names

        if scores is None or len(scores) == 0:
            self.log('ERROR: no validation metrics available to store')
            return

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

    def store_tsv_data(self, raw_data, name):
        file_path = path.join(self.results_path, '%s_data.tsv' % name)
        entry_id = 1
        entry_format = '%s\t%s\t%s\n'

        sentiment_mapping = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }

        with open(file_path, 'w+') as f:
            for d in raw_data:
                f.write('%s\n' % '\t'.join(d))

    def create_results_directories(self):
        '''This function is responsible for creating the results directory.'''
        if not path.isdir(self.results_path):
            os.makedirs(self.results_path)

    def log(self, msg):
        '''Simple logging function which only works in verbose mode.'''
        print('[%s] %s' % (self.name, msg))
