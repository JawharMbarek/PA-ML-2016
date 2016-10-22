#!/usr/bin/env python

import matplotlib.pyplot as plt
import getopt
import sys
import json

from utils import set_figure_size

set_figure_size()

metrics_file = ''
image_file = ''
only_list = []
nb_run = -1

try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:o:i:n:',
                               ['metrics=', 'image=', 'include='])
except getopt.GetoptError:
    print('./generate_metric_plot.py -m <metrics> -i <image.png>')
    print('                          -o <only-list> -n <run-number>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-m', '--metrics'):
        metrics_file = arg
    elif opt in ('-n', '--nb-run'):
        nb_run = arg
    elif opt in ('-i', '--image'):
        image_file = arg
    elif opt in ('-o', '--only'):
        only_list = arg.split(',')

with open(metrics_file, 'r') as f:
    metrics = json.loads(f.read())

metric_names = []

plt.title('metrics')
plt.xlabel('epoch')
plt.ylabel('%')

plt.ylim(0.0, 1.0)

if metrics_file.endswith('_all.json'):
    if nb_run == -1:
        print('Number of run to plot is missing (argument -n)')
        sys.exit(1)

    metrics = metrics[str(nb_run)]

for name, values in metrics.items():
    if name in only_list or len(only_list) == 0:
        metric_names.append(name)
        plt.plot(metrics[name])

plt.legend(metric_names, loc='upper left')

if image_file == '':
    plt.show()
else:
    plt.savefig(image_file)
