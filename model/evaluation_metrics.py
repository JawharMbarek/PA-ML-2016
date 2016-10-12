import numpy as np
from collections import Counter
import keras.backend as K
from  keras.utils.np_utils import to_categorical


def f1_score(y_true, y_pred, considered_classes=[0, 1, 2]):
    '''Computes the F1 score as usual, but only over the
       records with sentiments listed in considered classes.
       By default it uses three classes 0-2 corresponding to
       0=negative, 1=neutral, 2=positive.''' 

    # convert probas to 0,1
    y_ppred = K.zeros_like(y_true)
    y_pred_ones = K.T.set_subtensor(y_ppred[K.T.arange(y_true.shape[0]), K.argmax(y_pred, axis=-1)], 1)

    # where y_ture=1 and y_pred=1 -> true positive
    y_true_pred = K.sum(y_true * y_pred_ones, axis=0)

    # for each class: how many where classified as said class
    pred_cnt = K.sum(y_pred_ones, axis=0)

    # for each class: how many are true members of said class
    gold_cnt = K.sum(y_true, axis=0)

    # precision for each class
    precision = K.T.switch(K.T.eq(pred_cnt, 0), 0, y_true_pred / pred_cnt)

    # recall for each class
    recall = K.T.switch(K.T.eq(gold_cnt, 0), 0, y_true_pred / gold_cnt)

    # f1 for each class
    f1_class = K.T.switch(K.T.eq(precision + recall, 0), 0,
                          2 * (precision * recall) / (precision + recall))

    # return average f1 score over all classes
    return K.mean(f1_class[considered_classes])

def f1_score_pos(y_true, y_pred):
    '''Computes the F1 score as usual, but only over the
       records with a positive sentiment.'''
    return f1_score(y_true, y_pred, considered_classes=[2])

def f1_score_neg(y_true, y_pred):
    '''Computes the F1 score as usual, but only over the
       records with a negative sentiment.'''
    return f1_score(y_true, y_pred, considered_classes=[0])

def f1_score_pos_neg(y_true, y_pred):
    '''Computes the F1 score as usual, but it omits the
       neutral sentiment class and only considers the
       positives and negatives.'''
    return f1_score(y_true, y_pred, considered_classes=[0, 2])

def f1_score_neu(y_true, y_pred):
    '''Computes the F1 score as usual, but it only con-
       sideres the records with a neutral sentiments..'''
    return f1_score(y_true, y_pred, considered_classes=[1])