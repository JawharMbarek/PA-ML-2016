#!/usr/bin/env python

import json
import sys
import os.path as path
import numpy as np

from executor import Executor
from utils import generate_test_id

#
# Constants
#
RESULTS_PATH = path.join(path.dirname(path.realpath(__file__)), 'results')
RNG_SEED = 1337

#
# Basic setup
#
np.random.seed(RNG_SEED)

#
# Parameters for the run
#
vocabulary_path = ''
embeddings_path = ''
test_data_path = ''
validation_data_path = ''
verbose = False

#
# Argument handling
#
argv = sys.argv[1:]

if len(argv) == 0 or argv[0] == '':
    print("ERROR: JSON config is missing")
    print("       (e.g. ./run.sh config.json)")
    sys.exit(1)

for i in range(0, len(argv)):
    config_path = argv[i]
    print('Starting run with file %s' % config_path)

    with open(config_path) as f:
        params = json.loads(f.read())

    #
    # Execute the run!
    #
    test_id = generate_test_id(params)
    executor = Executor(test_id, params)
    executor.run()

    print('Finished run with file %s' % config_path)
