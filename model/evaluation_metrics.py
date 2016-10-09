import numpy as np
from collections import Counter
import keras.backend as K
from  keras.utils.np_utils import to_categorical


def f1_score_keras(y_true, y_pred):
    # convert probas to 0,1
    y_ppred = K.zeros_like(y_true)
    y_pred_ones = K.T.set_subtensor(y_ppred[K.T.arange(y_true.shape[0]), K.argmax(y_pred, axis=-1)], 1)

    # where y_ture=1 and y_pred=1 -> true positive
    y_true_pred = K.sum(y_true*y_pred_ones, axis=0)

    # for each class: how many where classified as said class
    pred_cnt = K.sum(y_pred_ones, axis=0)

    # for each class: how many are true members of said class
    gold_cnt = K.sum(y_true, axis=0)

    # precision for each class
    precision = K.T.switch(K.T.eq(pred_cnt, 0), 0, y_true_pred/pred_cnt)

    # recall for each class
    recall = K.T.switch(K.T.eq(gold_cnt, 0), 0, y_true_pred/gold_cnt)

    # f1 for each class
    f1_class = K.T.switch(K.T.eq(precision + recall, 0), 0, 2*(precision*recall)/(precision+recall))

    # return average f1 score over all classes
    return K.mean(f1_class)


def f1_score_keras_subj(y_true, y_pred):
    # convert probas to 0,1
    y_ppred = K.zeros_like(y_true)
    y_pred_ones = K.T.set_subtensor(y_ppred[K.T.arange(y_true.shape[0]), K.argmax(y_pred, axis=-1)], 1)

    y_true_0 = K.sum(y_true[:, [0]], axis=1).dimshuffle(0,'x')
    y_true_1 = K.sum(y_true[:, [1,2,3,4,5,6,7,8]], axis=1).dimshuffle(0,'x')

    y_pred_0 = K.sum(y_pred_ones[:, [0]], axis=1).dimshuffle(0,'x')
    y_pred_1 = K.sum(y_pred_ones[:, [1,2,3,4,5,6,7,8]], axis=1).dimshuffle(0,'x')

    y_true = K.concatenate([y_true_0, y_true_1], axis=1)
    y_pred = K.concatenate([y_pred_0, y_pred_1], axis=1)
    return f1_score_keras(y_true, y_pred)


if __name__ == '__main__':
    y_truth = np.array([0,0,0,1,2,0,1,2])
    y_pred =  np.array([2,0,2,2,0,2,2,2])

    print Counter(y_truth)
    print Counter(y_pred)

    y_true = K.variable(value=to_categorical(y_truth,3))
    y_pred = K.variable(value=to_categorical(y_pred,3))
    print K.eval(f1_score_keras(y_true, y_pred))
