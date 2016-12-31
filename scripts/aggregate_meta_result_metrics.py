#!/usr/bin/env python

import sys
import json
import time
import getopt
import os

from os import path

argv = sys.argv[1:]
out_file = 'aggregated_meta_validation_metrics_%d.csv'
header = ','.join([
    'name',
    'used domain models',
    'acc',
    'loss',
    'f1_score',
    'f1_score_pos_neg',
    'f1_score_pos',
    'f1_score_neg',
    'f1_score_neu',
    'train_data',
    'test_data\n'
])

metrics_line = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'

with open(out_file % int(time.time()), 'w+') as out_f:
    out_f.write(header)

    for group_dir in argv:
        result_dirs = map(lambda x: path.join(group_dir, x), os.listdir(group_dir))
        result_dirs = list(result_dirs)

        for result_dir in result_dirs:
            files = os.listdir(result_dir)
            files = map(lambda x: path.join(result_dir, x), files)
            files = list(files)

            config_file_path = None
            metrics_file_path = None

            config = {}
            metrics = {}

            for f in files:
                if f.endswith('.json'):
                    if '_config_' in f and config_file_path is None:
                        config_file_path = f
                    elif '_metrics_' in f and metrics_file_path is None:
                        metrics_file_path = f

            if config_file_path is None or metrics_file_path is None:
                print('Config or metrics missing in %s! Skipping...' % result_dir)
                continue

            name = result_dir
            used_domains = []

            with open(config_file_path, 'r') as config_f:
                config = json.loads(config_f.read())

            used_domains = config['used_domain_models']
            train_data = config['train_data_path']
            test_data = config['test_data_path']

            with open(metrics_file_path, 'r') as metrics_f:
                metrics = json.loads(metrics_f.read())

                out_f.write(metrics_line % (
                    name.split('/')[-1],
                    ' '.join(used_domains),
                    str(metrics.get('acc', '-')),
                    str(metrics.get('loss', '-')),
                    str(metrics.get('f1_score', '-')),
                    str(metrics.get('f1_score_pos_neg', '-')),
                    str(metrics.get('f1_score_pos', '-')),
                    str(metrics.get('f1_score_neg', '-')),
                    str(metrics.get('f1_score_neu', '-')),
                    train_data,
                    test_data
                ))

print('Aggregated all results from all experiments lying in %s' % ', '.join(argv))