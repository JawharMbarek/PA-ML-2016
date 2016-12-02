#!/usr/bin/env python

import sys
import numpy as np

argv = sys.argv[1:]

test_perc = 0.2

if len(argv) != 4:
    print('ERROR: missing mandatory parameters')
    print('       python scripts/split_tsv_file.py <in-tsv> <out-train-tsv> <out-validation-tsv> <perc-validation>')
    sys.exit(2)

in_tsv = argv[0]
out_train_tsv = argv[1]
out_validation_tsv = argv[2]
perc_validation = float(argv[3])

with open(in_tsv, 'r') as in_f:
    in_entries = in_f.read().split('\n')
    nr_of_entries = len(in_entries)

    # in case the last line is empty reduced the nr_of_entries count by one
    if in_entries[-1] == '':
        nr_of_entries -= 1

    nr_of_validation_entries = int(float(nr_of_entries) * perc_validation)
    selected_validation_idxs = []

    while len(selected_validation_idxs) < nr_of_validation_entries:
        idx = np.random.randint(0, nr_of_validation_entries)

        if idx in selected_validation_idxs:
            continue
        else:
            selected_validation_idxs.append(idx)

    with open(out_train_tsv, 'w+') as out_train_f:
        with open(out_validation_tsv, 'w+') as out_validation_f:
            for i, line in enumerate(in_f):
                if i in selected_validation_idxs:
                    out_validation_f.write(line)
                else:
                    out_train_f.write(line)

print('Splitted the TSV file %s into %s and %s by the ratio %.1f%% / %.1f%%' % (
    in_tsv, out_train_tsv, out_validation_tsv, (1.0 - perc_validation) * 100.0, perc_validation * 100.0
))
