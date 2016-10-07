import numpy as np
import parse_utils
import re

from keras import backend as K

polarity_task_conversion = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}


def evalitalia_loader(fname,alphabet,tknzr, delimiter='\t'):
    dummy_word_idx = alphabet.get('DUMMY_WORD_IDX', 1)
    print("alphabet", len(alphabet))
    print('dummy_word:', dummy_word_idx)

    data_raw = open(fname, 'rb').readlines()
    data_raw = list(map(lambda x: x.decode('utf-8').replace('\n', '').split(delimiter), data_raw))
    #tids = np.fromiter(map(lambda x: x[0], data_raw), dtype=np.int)
    lables = np.fromiter(map(lambda x: polarity_task_conversion.get(x[-2]), data_raw), dtype=np.int)
    nlabels = 3

    tweets = map(lambda x: tknzr.tokenize(parse_utils.preprocess_tweet(x[-1])), data_raw)
    tweet_idx = parse_utils.convert2indices(tweets, alphabet, dummy_word_idx)

    return [], lables, tweet_idx, nlabels


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

# F1 score for keras
def fbeta_score(y_true, y_pred, beta=1):
    '''Compute F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification where input samples can be
    tagged with a set of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score
