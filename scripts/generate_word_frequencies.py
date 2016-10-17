#!/usr/bin/env python

import os
import numpy as np
import sys
import time
import pickle

frequency_dict = {}
output_file = 'word_frequencies_%d.pickle' % int(time.time())
count_new = 0
count = 0

for file in sys.argv[1:]:
    with open(file, 'r') as f:
        print('Start processing file %s...' % file)

        for line in f:
            for word in line.split():
                if word not in frequency_dict:
                    frequency_dict[word] = 0
                    count_new += 1

                frequency_dict[word] += 1
                count += 1

        print('Finished processing file %s...' % file)
        print('(%d words processed, %d new words found)' % (count, count_new))

        count_new = 0

with open(output_file, 'wb+') as f:
    pickle.dump(frequency_dict, f)

print('Saved frequency dictionary to %s' % output_file)
