#!/usr/bin/env python

import matplotlib.pyplot as plt
import getopt
import sys
import json

metrics_file = ''
image_file = ''
only_list = []

try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:o:i:',
                               ['metrics=', 'image=', 'include='])
except getopt.GetoptError:
    print('./generate_metric_plot.py -m <metrics> -i <image.png>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-m', '--metrics'):
        metrics_file = arg
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

for name, values in metrics.items():
    if name in only_list or len(only_list) == 0:
        metric_names.append(name)
        plt.plot(metrics[name])

plt.legend(metric_names, loc='upper left')

if image_file == '':
    plt.show()
else:
    plt.savefig(image_file)
