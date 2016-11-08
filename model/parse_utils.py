import numpy as np
import re

UNKNOWN_WORD_IDX = 0
URL_REGEX = '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))'


def preprocess_tweet(tweet):
    # lowercase and normalize urls
    tweet = tweet.replace('\n', '').lower()
    tweet = re.sub(URL_REGEX, '<url>', tweet)
    tweet = re.sub('@[^\s]+', '<user>', tweet)

    # tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


def convert2indices(data, alphabet, dummy_word_idx,
                    max_sent_length=140, verbose=0):
    data_idx = []
    max_len = 0
    unknown_words = 0

    if type(data) is list and type(data[0]) is str:
        data = [data]

    for sentence in data:
        ex = np.ones(max_sent_length) * dummy_word_idx
        max_len = max(len(sentence), max_len)

        if len(sentence) > max_sent_length:
            sentence = sentence[:max_sent_length]

        for i, token in enumerate(sentence):
            idx = alphabet.get(token, UNKNOWN_WORD_IDX)
            ex[i] = idx

            if idx == UNKNOWN_WORD_IDX:
                unknown_words += 1

        data_idx.append(ex)

    data_idx = np.array(data_idx).astype('int32')

    if verbose == 1:
        print("Max length in this batch:", max_len)
        print("Number of unknown words:", unknown_words)

    return data_idx
