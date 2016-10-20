#!/usr/bin/env python

import json
import sys
import os
import numpy as np
import time

from os import path
from executor import Executor
from utils import generate_test_id
from subprocess import Popen, PIPE

#
# Constants
#
RESULTS_PATH = path.join(path.dirname(path.realpath(__file__)), 'results')

#
# Parameters for the run
#
vocabulary_path = ''
embeddings_path = ''
test_data = ''
validation_data_path = ''
verbose = False
git_sha = ''
np_rand_seed = int(time.time())

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
    configs = []

    if path.isdir(config_path):
        for c in os.listdir(config_path):
            if c.endswith('.json'):
                configs.append(path.join(config_path, c))
    else:
        configs = [config_path]

    print('The following configs will be run:\n* %s' % '\n* '.join(configs))

    for cfg in configs:
        print('Starting run with file %s' % cfg)

        with open(cfg) as f:
            params = json.loads(f.read())

        # Take the config name for the results directory
        # in case of no name is defined
        if not 'name' in params:
            params['name'] = path.splitext(path.basename(cfg))[0]

        # Store the git hash in the params
        params['git_rev'] = git_rev

        if 'np_rand_seed' not in params:
            params['np_rand_seed'] = np_rand_seed
        else:
            np_rand_seed = params['np_rand_seed']

        np.random.seed(np_rand_seed)

        #
        # Execute the run!
        #
        test_id = generate_test_id(params)
        executor = Executor(test_id, params)
        executor.run()

        print('Finished run with file %s' % config_path)
