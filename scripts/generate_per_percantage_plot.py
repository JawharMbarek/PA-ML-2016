import matplotlib.pyplot as plt
import getopt
import sys
import re
import os
import json
import numpy as np

from os import path

absolute_values = False
results_path = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], 'a:r:',
                               ['results=', 'absolute'])
except getopt.GetoptError:
    print('./generate_per_percentage_plot.py <results-directory> (-a)')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-a', '--absolute'):
        absolute_values = True
    elif opt in ('-r', '--results'):
        results_path = arg

base_count  = 8000
valid_count = 7000

x_data = []
y_data = []

for perc in range(0, 10):
    perc *= 10
    perc_str = str('%dPercent' % perc)

    for dir in os.listdir(results_path):
        if perc_str in dir:
            print('Loading data for %s (for %f%%)' % (dir, perc))

            metrics_path = path.join(results_path, dir, 'validation_metrics.json')

            with open(metrics_path) as f:
                metrics = json.load(f)

                x_data.append(float(perc) / 100.0)
                y_data.append(np.mean([
                    metrics['all'][i]['f1_score_pos_neg'] for i in range(0, 10)
                ]))


plt.plot(x_data, y_data)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.ylabel('F1 Score (Pos/Neg)')
plt.xlabel('Percentage of domain specific training data')
plt.show()
