import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import path

from utils import set_figure_size

set_figure_size()

SENTIMENT_COLOR = {
    'neutral': 'brown',
    'positive': 'green',
    'negative': 'red'
}

argv = sys.argv[1:]
stats = None

if len(argv) == 0:
    print('ERROR: Missing required text statistics file parameter')
    print('       python scripts/generate_text_statistics_plot.py <tex-statistics>')
    sys.exit(2)

stats_path = argv[0]
stats_files = []

with open(stats_path, 'r') as stats_file:
    stats = json.load(stats_file)['data']

stats_files = stats.keys()
stats_files = []
stats_labels = ['negative', 'neutral', 'positive']

with sns.axes_style('white'):
    sns.set_style('ticks')
    sns.set_context('talk')

    # plot details
    bar_width = 0.35
    epsilon = .015
    line_width = 1
    opacity = 0.7
    pos_bar_positions = np.arange(len(stats))
    neu_bar_positions = pos_bar_positions + bar_width
    neg_bar_positions = pos_bar_positions + (2 * bar_width)

    all_pos_percentages = []
    all_neu_percentages = []
    all_neg_percentages = []

    for f, d in stats.items():
        stats_files.append(path.basename(f))

        all_pos_percentages.append(d['avg']['sentiment_pos_percentage'])
        all_neu_percentages.append(d['avg']['sentiment_neu_percentage'])
        all_neg_percentages.append(d['avg']['sentiment_neg_percentage'])

    plt.bar(pos_bar_positions, all_pos_percentages, bar_width, color=SENTIMENT_COLOR['positive'], label='Positive')
    plt.bar(neu_bar_positions, all_neu_percentages, bar_width, color=SENTIMENT_COLOR['neutral'], label='Neutral')
    plt.bar(neg_bar_positions, all_neg_percentages, bar_width, color=SENTIMENT_COLOR['negative'], label='Negative')

plt.legend(loc='best')
plt.xlabel('Data files')
plt.ylabel('Percentage of samples')
plt.xticks(pos_bar_positions + bar_width, stats_files, rotation=90, fontsize=8)
plt.ylim(0.0, 1.0)
plt.show()
