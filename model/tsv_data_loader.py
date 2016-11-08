import numpy as np
import parse_utils

from data_utils import tsv_sentiment_loader
from nltk import TweetTokenizer
from keras.utils.np_utils import to_categorical

class TsvDataLoader(object):
    SENTIMENT_MAP = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }

    '''This class can be used to load multiple TSV files
       with sentiments and shuffle them in ratios provided
       by the user through the config (if necessary).'''
    def __init__(self, str_or_obj, vocabulary,
                 randomize=False, max_sent_length=500):
        self.randomize = randomize
        self.max_sent_length = max_sent_length
        self.vocabulary = vocabulary

        if type(str_or_obj) is str:
            self.path = str_or_obj
        else:
            self.obj = str_or_obj

    def load_lazy(self):
        curr_count = 0

        with open(self.path, 'r') as f:
            for line in f:
                yield self.process_line(line)
                curr_count += 1

                if curr_count % 100000:
                    print('Processed %d training examples' % curr_count)

        curr_count = 0

    def load(self):
        '''Loads the data in one of two ways: If the first param is a
           string, it tries to interpret it as a path and load the file
           it is pointing to. If it is an object, it treats it as a dict
           where the keys are the paths and the values are the ratios in
           which they should appear in the resulting dataset.'''

        if type(self.str_or_obj) is str:
            return self.load_from_path()
        else:
            return self.load_from_object()

    def load_from_path(self):
        '''Loads the TSV file at the given path.''' 
        tknzr = self.get_tokenizer()
        tids, sentiments, texts, raw_data, nlabels = tsv_sentiment_loader(self.str_or_obj, self.vocabulary, tokenizer)
        return (sentiments, texts, raw_data, nlabels)

    def process_line(self, line):
        dummy_word_idx = self.vocabulary.get('DUMMY_WORD_IDX', 1)
        tknzr = self.get_tokenizer()

        line_data = line.replace('\n', '').split('\t')
        sentiment = self.SENTIMENT_MAP[line_data[-2]]
        text = parse_utils.preprocess_tweet(line_data[-1])
        text_tokens = tknzr.tokenize(text)
        text_idxs = parse_utils.convert2indices(
            text_tokens, self.vocabulary, dummy_word_idx,
            max_sent_length=self.max_sent_length
        )

        return (np.asarray(text_idxs[0]), np.int(sentiment))

    def get_tokenizer(self):
        ''' Returns an instance of a tokenizer.'''
        return TweetTokenizer(reduce_len=True)
