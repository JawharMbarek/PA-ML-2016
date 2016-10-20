#!/usr/bin/env python

import matplotlib.pyplot as plt
import getopt
import sys
import json
import re
import os

from os import path

argv = sys.argv[1:]
metrics_mappings = {}
metrics_names = []
combined_metrics_re = re.compile('^.+={1}[a-zA-Z0-9_,]+$')
count = 1

if len(argv) == 0:
    print('./generate_combine_metrics_plot.py <metrics-map>')
    sys.exit(2)

for m in argv:
    if combined_metrics_re.match(m) != None:
        metrics_file = m.split('=')[0]
        metrics_file_base = path.join(
            path.basename(path.dirname(metrics_file)),
            path.basename(metrics_file)
        )

        curr_metrics_names = m.split('=')[1].split(',')
        metrics = None

        with open(metrics_file, 'r') as f:
            metrics = json.loads(f.read())

        for name, values in metrics.items():
            if name in curr_metrics_names or len(curr_metrics_names) == 0:
                metrics_names.append('%s (%s)' % (name, metrics_file_base))
                plt.plot(metrics[name])
    else:
        print('invalid metrics map entry: %s' % m)


plt.title('metrics')
plt.xlabel('epoch')
plt.ylabel('%')

plt.ylim(0.0, 1.0)

plt.legend(metrics_names, loc='best')

plt.show()
