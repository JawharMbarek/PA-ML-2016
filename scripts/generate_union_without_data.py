#!/usr/bin/env python

import os
import sys

from os import path

data_path = path.abspath(path.join(path.dirname(__file__), '..', 'testdata'))

tsv_files = [
    'DAI_tweets_test.tsv',
    'DIL_reviews_test.tsv',
    'HUL_reviews_test.tsv',
    'JCR_quotations_test.tsv',
    'MPQ_reviews_test.tsv',
    'SEM_headlines_test.tsv',
    'SemEval_tweets_test.tsv',
    'TAC_reviews_test.tsv'
]

union_file_pattern = 'union_without_%s_test.tsv'

tsv_data = {}
domains = []

def get_domain(name):
    return name.split('_')[0].lower()

for file in tsv_files:
    file_path = path.join(data_path, file)
    domain_name = get_domain(file)

    with open(file_path, 'r') as f:
        tsv_data[domain_name] = f.read().split('\n')

for domain in tsv_data.keys():
    file_path = path.join(data_path, union_file_pattern % domain)
    file_data = []

    for other_domain, data in tsv_data.items():
        if domain == other_domain:
            continue
        else:
            file_data += data

    with open(file_path, 'w+') as f:
        for d in file_data:
            f.write('%s\n' % d)

print('Finished creating all union_without_XYZ datasets')
