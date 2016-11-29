#!/usr/bin/env python

import sys
import os
import json

from os import path

argv = sys.argv[1:]

if len(argv) < 6:
    print('ERROR: missing mandatory arguments')
    print('       python scripts/generate_albation_experiments.py <embeddings> <vocabulary>')
    print('                                                       <model-json> <model-weights>')
    print('                                                       <embedings-name> <tsv1,tsv2, ...>')
    sys.exit(2)

embeddings_path = argv[0]
vocabulary_path = argv[1]
model_json_path = argv[2]
model_weights_path = argv[3]
embeddings_name = argv[4]
tsv_files = argv[5:]

configs_path = path.abspath(path.join(path.dirname(__file__), '..', 'configs'))
albation_configs_path = path.join(configs_path, 'crossdomain_general_albation')

group_id = 'crossdomain_general_albation_%s_embeddings' % embeddings_name

config_template = {
    'group_id': group_id,
    'set_class_weights': True,
    'test_data': {},
    'nb_epoch': 1000,
    'nb_kfold_cv': 1,
    'vocabulary_embeddings': None,
    'validation_data_path': '',
    'validate_while_training': False,
    'vocabulary_path': None,
    'model_json_path': None,
    'model_weights_path': None
}

if not path.isdir(albation_configs_path):
    print('Creating configs directory %s because it is missing' % albation_configs_path)
    os.mkdir(albation_configs_path)

tsv_sizes = {}

for tsv_file in tsv_files:
    with open(tsv_file, 'r') as f:
        tsv_sizes[tsv_file] = len(f.read().split('\n')) - 1

for tsv in tsv_sizes.keys():
    tsv_name = tsv.split('/')[-1].replace('.tsv', '').lower()
    config_name = '%s_%s' % (group_id, tsv_name)
    config_path = '%s.json' % path.join(albation_configs_path, config_name)

    if path.isfile(config_path):
        print('ERROR: The config %s already exists, exiting' % config_path)
        sys.exit(2)

    curr_config = config_template.copy()
    curr_config['validation_data'] = tsv
    curr_config['vocabulary_embeddings'] = embeddings_path
    curr_config['vocabulary_path'] = vocabulary_path
    curr_config['model_json_path'] = model_json_path
    curr_config['model_weights_path'] = model_weights_path

    with open(config_path, 'w+') as f:
        for other_tsv, size in tsv_sizes.items():
            curr_config['test_data'][other_tsv] = size

        f.write(json.dumps(curr_config, indent=2,
                           separators=(',', ': '), sort_keys=True))

print('Created albation experiment configs for all supplied TSV files')