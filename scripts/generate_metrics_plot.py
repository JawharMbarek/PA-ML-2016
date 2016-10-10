#!/usr/bin/env python

import matplotlib.pyplot as plt
import getopt
import sys
import json

metrics_file = ''
output_file = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:o:', ['metrics=', 'output='])
except getopt.GetoptError:
    print('./generate_metric_plot.py -m <metrics> -o <output.png>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-m', '--metrics'):
        metrics_file = arg
    elif opt in ('-o', '--output'):
        output_file = arg

with open(metrics_file, 'r') as f:
    metrics = json.loads(f.read())

metric_names = []

plt.title('metrics')
plt.xlabel('epoch')
plt.ylabel('%')

for name, values in metrics.items():
    metric_names.append(name)
    plt.plot(metrics[name])

plt.legend(metric_names, loc='upper left')

if output_file == '':
    plt.show()
else:
    plt.savefig(output_file)
