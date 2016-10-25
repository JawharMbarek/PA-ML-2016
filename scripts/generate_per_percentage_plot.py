#!/usr/bin/env python

import matplotlib.pyplot as plt
import getopt
import sys
import re
import os
import json
import numpy as np

from os import path
from utils import set_figure_size

set_figure_size()

image_path = ''
absolute_values = False
results_path = ''
only_metrics = 'val_f1_score_pos_neg,f1_score_pos_neg'

try:
    opts, args = getopt.getopt(sys.argv[1:], 'a:r:i:m:',
                               ['results=', 'only_metrics=', 'image=', 'absolute'])
except getopt.GetoptError:
    print('./generate_per_percentage_plot.py -r <results-path> -m <only_metrics>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-a', '--absolute'):
        absolute_values = True
    elif opt in ('-r', '--results'):
        results_path = arg.split(',')
    elif opt in ('-i', '--image'):
        image_path = arg
    elif opt in ('-m', '--only-metrics'):
        only_metrics = arg.split(',')

base_count  = 8000
valid_count = 7000

x_data = []
y_data = {}

only_metrics = list(set([x.replace('val_', '') for x in only_metrics]))

already_loaded = {}

for perc in range(0, 21):
    perc *= 10
    perc_str = str('_%dPercent' % perc)

    for rpath in results_path:
        group_name = rpath.replace('results/', '')

        abs_perc = float(perc) / 100.0

        if not abs_perc in x_data:
            x_data.append(abs_perc)

        if not group_name in y_data:
            y_data[group_name] = {}

        if not group_name in already_loaded or not perc_str in already_loaded[group_name]:
            already_loaded[group_name] = {perc_str: False}

        for dir in os.listdir(rpath):
            if perc_str in dir and not already_loaded[group_name][perc_str]:
                print('Loading data for %s (for %f%%)' % (dir, perc))

                metrics_path = path.join(rpath, dir, 'validation_metrics.json')

                with open(metrics_path) as f:
                    metrics = json.load(f)
                    stop_idx = len(metrics['all'])

                    for m in only_metrics:
                        if not m in y_data[group_name]:
                            y_data[group_name][m] = []

                        y_data[group_name][m].append(np.mean([metrics['all'][i][m] for i in range(0, stop_idx)]))

                already_loaded[group_name][perc_str] = True

keys = []
max_entries_length = -1

for r, m in y_data.items():
    for k, values in m.items():
        if len(values) > max_entries_length:
            max_entries_length = len(values)

        keys.append('%s (%s)' % (k, r))
        plt.plot(x_data[0:len(values)], values)

plt.xlim(0.0, (max_entries_length - 1) * 0.1)
plt.ylim(0.0, 1.0)
plt.ylabel(', '.join(only_metrics))
plt.xlabel('Percentage of domain specific training data')

plt.legend(keys, loc='best')

if image_path == '':
    plt.show()
else:
    plt.savefig(image_path)
