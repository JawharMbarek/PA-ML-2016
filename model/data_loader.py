import numpy as np

from data_utils import tsv_sentiment_loader
from nltk import TweetTokenizer

class DataLoader(object):
    '''This class can be used to load multiple TSV files
       with sentiments and shuffle them in ratios provided
       by the user through the config (if necessary).'''

    @staticmethod
    def load(str_or_obj, vocabulary):
        '''Loads the data in one of two ways: If the first param is a
           string, it tries to interpret it as a path and load the file
           it is pointing to. If it is an object, it treats it as a dict
           where the keys are the paths and the values are the ratios in
           which they should appear in the resulting dataset.'''

        if type(str_or_obj) is str:
            return DataLoader.load_from_path(str_or_obj, vocabulary)
        else:
            return DataLoader.load_from_object(str_or_obj, vocabulary)

    @staticmethod
    def load_from_path(path, vocabulary):
        '''Loads the TSV file at the given path.''' 
        tokenizer = TweetTokenizer(reduce_len=True)
        tids, sentiments, texts, nlabels = tsv_sentiment_loader(path, vocabulary, tokenizer)
        return (sentiments, texts, nlabels) # leave out TIDS for now

    @staticmethod
    def load_from_object(obj, vocabulary):
        '''Loads different datasets with given ratios to create a new
           dataset which contains as much records from the each file
           as specified.'''

        tmp_texts = []
        tmp_sentiments = []

        curr_nlabels = -1

        # load all data and count how much records there are
        for path, count in obj.items():
            sentiments, texts, nlabels = DataLoader.load_from_path(path, vocabulary)

            if curr_nlabels == -1:
                curr_nlabels = nlabels
            elif curr_nlabels != nlabels:
                raise Exception('cannot handle differnt count of labels (%d != %d)'
                                % (curr_nlabels, nlabels))

            records_count = len(sentiments)
            sel_sentiments = []
            sel_texts = []
            sel_idx = []

            # in case we've too few data, we simply cut down the count and
            # print a warning to the terminal
            if count > records_count:
                print('WARNING: Do not have enough records in %s (%d < %d)'
                      % (path, records_count, count))

                count = records_count

            # randomly select 'count' much records from the loaded data
            while len(sel_idx) < count:
                curr_idx = -1

                # find idx which we haven't already picked
                while curr_idx == -1 or curr_idx in sel_idx:
                    curr_idx = np.random.randint(0, records_count)

                sel_sentiments.append(sentiments[curr_idx])
                sel_texts.append(texts[curr_idx])
                sel_idx.append(curr_idx)

            tmp_sentiments += sel_sentiments
            tmp_texts += sel_texts

        res_texts = []
        res_sentiments = []

        rng_idx_list = list(range(0, len(tmp_sentiments)))

        # shuffle the resulting dataset
        for idx in rng_idx_list:
            res_texts.append(tmp_texts[idx])
            res_sentiments.append(tmp_sentiments[idx])

        return np.asarray(res_sentiments), np.asarray(res_texts), curr_nlabels
