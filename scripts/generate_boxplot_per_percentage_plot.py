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

#set_figure_size()

image_path = ''
absolute_values = False
results_path = ''
metric = 'f1_score_pos_neg'

try:
    opts, args = getopt.getopt(sys.argv[1:], 'a:r:i:m:',
                               ['results=', 'metric=', 'image=', 'absolute'])
except getopt.GetoptError:
    print('./generate_boxplot_per_percentage_plot.py -r <results-path> -m <metric>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-a', '--absolute'):
        absolute_values = True
    elif opt in ('-r', '--results'):
        results_path = arg
    elif opt in ('-i', '--image'):
        image_path = arg
    elif opt in ('-m', '--metric'):
        metric = arg

base_count  = 8000
valid_count = 7000

x_data = []
y_data = {}

for perc in range(0, 21):
    perc *= 10
    perc_str = str('_%dPercent' % perc)
    already_loaded = False
    x_data.append(float(perc) / 100.0)

    for dir in os.listdir(results_path):
        if perc_str in dir and not already_loaded:
            print('Loading data for %s (for %f%%)' % (dir, perc))

            metrics_path = path.join(results_path, dir, 'train_metrics_opt.json')

            with open(metrics_path) as f:
                metrics = json.load(f)
                y_data[perc] = metrics[metric]

            already_loaded = True

keys = []

plt.boxplot(list(y_data.values()), patch_artist=True)
plt.legend(keys, loc='best')
plt.ylim(0.0, 2.0)

plt.xlabel('Boxplots of %s for 10%%-200%% domain specific data' % metric)

if image_path == '':
    plt.show()
else:
    plt.savefig(image_path)
