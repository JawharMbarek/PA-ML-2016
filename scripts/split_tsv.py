#!/usr/bin/env python

import sys

argv = sys.argv[1:]

test_perc = 0.2

if len(argv) != 3:
    print('ERROR: missing mandatory parameters')
    print('       python scripts/split_tsv_file.py <in-tsv> <out-train-tsv> <out-test-tsv>')
    sys.exit(2)