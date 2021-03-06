import numpy as np
import parse_utils

from keras import backend as K
from itertools import chain, combinations

POLARITY_TASK_CONVERSION = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

def compute_class_weights(sentiments):
    '''Computes the class weights for the given list of sentiments.'''
    class_weights = {}
    tmp_class_weights = {}
    nb_per_class = len(sentiments) / 3

    for s in sentiments:
        if not s in tmp_class_weights:
            tmp_class_weights[s] = 0

        tmp_class_weights[s] += 1

    for k, n in tmp_class_weights.items():
        class_weights[k] = nb_per_class / n

    return class_weights

def tsv_sentiment_loader(fname, alphabet, tokenizer, delimiter='\t'):
    '''This function loads a TSV file containing example texts with
       their corresponding sentiments.'''

    dummy_word_idx = alphabet.get('DUMMY_WORD_IDX', 1)

    data_raw = open(fname, 'rb').readlines()
    data_raw = filter(lambda x: len(x.split()) > 0, data_raw)
    data_raw = map(
        lambda x: x.decode('utf-8').replace('\n', '').split(delimiter),
        data_raw
    )

    data_raw = list(data_raw)

    # TODO: What to do with broken tids?
    # tids = np.fromiter(map(lambda x: x[0], data_raw), dtype=np.int)

    lables = np.fromiter(
        map(lambda x:
            POLARITY_TASK_CONVERSION.get(x[-2], POLARITY_TASK_CONVERSION['neutral']),
            data_raw), dtype=np.int
    )

    nlabels = 3

    tweets = map(
        lambda x: tokenizer.tokenize(parse_utils.preprocess_tweet(x[-1])),
        data_raw
    )

    tweet_idx = parse_utils.convert2indices(tweets, alphabet, dummy_word_idx)

    return [], lables, tweet_idx, data_raw, nlabels

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

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

    # count positive examples
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # set the f score to 0 if there's no positive example
    if c3 == 0:
        return 0

    # how many selected items are relevant?
    precision = c1 / c2

    # how many relevant items are selected?
    recall = c1 / c3

    # weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)

    return f_score
