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
        results_path = arg
    elif opt in ('-i', '--image'):
        image_path = arg
    elif opt in ('-m', '--only-metrics'):
        only_metrics = arg.split(',')

base_count  = 8000
valid_count = 7000

x_data = []
y_data = {}

only_metrics = list(set([x.replace('val_', '') for x in only_metrics]))

for perc in range(0, 21):
    perc *= 10
    perc_str = str('_%dPercent' % perc)
    already_loaded = False

    for dir in os.listdir(results_path):
        if perc_str in dir and not already_loaded:
            print('Loading data for %s (for %f%%)' % (dir, perc))

            metrics_path = path.join(results_path, dir, 'validation_metrics.json')

            with open(metrics_path) as f:
                metrics = json.load(f)
                stop_idx = len(metrics['all'])
                x_data.append(float(perc) / 100.0)

                for m in only_metrics:
                    if not m in y_data:
                        y_data[m] = []

                    y_data[m].append(np.mean([
                        metrics['all'][i][m] for i in range(0, stop_idx)
                    ]))

            already_loaded = True

keys = []

for k, m in y_data.items():
    plt.plot(x_data, m)
    keys.append(k)

plt.xlim(0.0, 2.0)
plt.ylim(0.0, 1.0)
plt.ylabel(', '.join(keys))
plt.xlabel('Percentage of domain specific training data')

plt.legend(keys, loc='best')

if image_path == '':
    plt.show()
else:
    plt.savefig(image_path)
