#!/usr/bin/env python

import os
import json

from os import path

configs_path = path.abspath(path.join(path.dirname(__file__), '..', 'configs'))
failed_config_paths = []
configs_to_check = []
dirs_to_check = [configs_path]

def try_load_json(file_path):
    success = False

    try:
        with open(file_path, 'r') as f:
            json.load(f)
            success = True
    except json.decoder.JSONDecodeError as e:
        success = False

    return success

print('Starting scan to find all configs...')

while len(dirs_to_check) > 0:
    curr_dir = dirs_to_check.pop()

    if not path.isdir(curr_dir):
        print('Skipping path %s, does not seem to be a directory' % curr_dir)
        continue

    for f in os.listdir(curr_dir):
        curr_full_path = path.join(curr_dir, f)

        if path.isdir(curr_full_path):
            dirs_to_check.append(curr_full_path)
        elif curr_full_path.endswith('.json'):
            configs_to_check.append(curr_full_path)

print('Finished collecting all configs!')
print('Starting to validate all found configs...')

for i, cfg in enumerate(configs_to_check):
    if not try_load_json(cfg):
        failed_config_paths.append(cfg)

    if (i+1) % 100 == 0:
        print('Checked %d of %d configs...' % ((i+1), len(configs_to_check)))

print('Finished validating all found configs!')

if len(failed_config_paths) > 0:
    err_configs = map(lambda x: '/'.join(x.split('/')[-2:]), failed_config_paths)
    err_configs = list(map(lambda x: '- %s' % x, err_configs))
    print('The following JSON configurations could not be loaded due to syntax errors:')
    print('')
    print('\n'.join(err_configs))
else:
    print('Success! No invalid JSON configurations were found.')
