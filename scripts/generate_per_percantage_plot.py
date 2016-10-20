import matplotlib.pyplot as plt
import getopt
import sys
import re
import os
import json
import numpy as np

from os import path

pattern_re = re.compile('[a-zA-Z_]+XX.*')

argv = sys.argv[1:]
name = argv[0]

if not pattern_re.match(name):
      print('Error with name!')
      sys.exit(2)

base_count  = 8000
valid_count = 7000

x_data = []
y_data = []

for perc in range(1, 10):
    perc *= 10

    print('Loading data for %s' % name.replace('XX', str(perc)))

    name_pattern = name.replace('XX', str(perc))
    dir_pattern = re.compile('.*%s.*' % name_pattern)

    for dir in os.listdir('results/'):
        if dir_pattern.match(dir):
            metrics_path = path.join('results', dir, 'validation_metrics.json')

            with open(metrics_path) as f:
                metrics = json.load(f)

                x_data.append(float(perc) / 100.0)
                y_data.append(np.mean([
                    metrics['all'][i]['f1_score_pos_neg'] for i in range(0, 10)
                ]))


plt.plot(x_data, y_data)
plt.ylabel('F1 Score (Pos/Neg)')
plt.xlabel('Percentage of domain specific training data')
plt.show()
