#!/usr/bin/env python

import sys
import json
import time
import getopt

from os import path

argv = sys.argv[1:]
out_file = 'aggregated_validation_metrics_%d.csv'
scope = 'all'

header = ','.join([
    'name',
    '#kfold',
    'acc',
    'loss',
    'f1_score',
    'f1_score_pos_neg',
    'f1_score_pos',
    'f1_score_neg',
    'f1_score_neu\n'
])

metrics_line = '%s,%d,%f,%f,%f,%f,%f,%f,%f\n'

with open(out_file % int(time.time()), 'w+') as f:
    f.write(header)

    nb_kfold = -1 # is set after loading the first
                  # set of metrics

    for results_dir in argv:
        metrics_file = path.join(results_dir, 'validation_metrics.json')
        params_file = path.join(results_dir, 'params.json') 

        name = ''

        with open(params_file, 'r') as params_f:
            params = json.loads(params_f.read())
            name = params['name']

        if name == '':
            name = 'NAME_NOT_FOUND'

        with open(metrics_file, 'r') as metrics_f:
            metrics = json.loads(metrics_f.read())

            all_metrics = metrics['all']
            avg_metrics = metrics['avg']

            if nb_kfold == -1:
                nb_kfold = len(all_metrics)

            for m in all_metrics:
                f.write(metrics_line % (
                    name,
                    m['round'],
                    m['acc'],
                    m['loss'],
                    m['f1_score'],
                    m['f1_score_pos_neg'],
                    m['f1_score_pos'],
                    m['f1_score_neg'],
                    m['f1_score_neu']
                ))
