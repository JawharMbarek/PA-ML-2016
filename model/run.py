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

config_path = argv[0]

with open(config_path) as f:
    params = json.loads(f.read())

#
# Basic setup
#
np.random.seed(RNG_SEED)

#
# Execute the run!
#
test_id = generate_test_id(params)
executor = Executor(test_id, params)
executor.run()
