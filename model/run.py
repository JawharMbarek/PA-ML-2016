#!/usr/bin/env python

import json
import sys
import os.path as path
import numpy as np

from executor import Executor
from utils import generate_test_id
from subprocess import Popen, PIPE

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
git_sha = ''

#
# Argument handling
#
argv = sys.argv[1:]

if len(argv) == 0 or argv[0] == '':
    print("ERROR: JSON config is missing")
    print("       (e.g. ./run.sh config.json)")
    sys.exit(1)

# Try to load the SHA of the current git revision or error out
git_proc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, stderr=PIPE)

git_out = git_proc.communicate()[0][0:-1] # cut off \n and second line
git_rev = git_out.decode('utf-8')
git_err = git_proc.returncode

if git_err != 0:
    print('ERROR: error while fetching current git revision SHA')
    sys.exit(1)

for i in range(0, len(argv)):
    config_path = argv[i]
    print('Starting run with file %s' % config_path)

    with open(config_path) as f:
        params = json.loads(f.read())

    # Store the git hash in the params
    params['git_rev'] = git_rev

    #
    # Execute the run!
    #
    test_id = generate_test_id(params)
    executor = Executor(test_id, params)
    executor.run()

    print('Finished run with file %s' % config_path)
