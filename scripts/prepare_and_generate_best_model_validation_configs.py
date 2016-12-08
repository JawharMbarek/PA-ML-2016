#!/usr/bin/env python

import os
import sys
import json
import shutil

from os import path

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: Missing mandatory arguments')
    print('       python scripts/prepare_and_generate_best_model_validation_configs.py <result-dirs>')

if not path.isdir('.git'):
    print('ERROR: This script can only be executed from the root of the project')
    sys.exit(2)

monitored_metric = 'f1_score_pos_neg'

valid_file_name = 'validation_metrics.json'
weights_file_name = 'weights_1.h5'
params_file_name = 'params.json'
model_file_name = 'model.json'

group_name_pattern = 'crossdomain_we_ds_'

models_dir = 'models/'
models_dir_pattern = 'best_model_crossdomain_we_ds_%s'
configs_dir_pattern = 'configs/crossdomain_sp_%s'
configs_file_pattern = 'crossdomain_sp_%s_%s.json'
group_id_pattern = 'crossdomain_sp_%s'

domain_names = set()
domain_files = set()

params_per_domain = {}
results_per_domain = {}
valid_data_per_domain = {}
best_result_directory_per_domain = {}
best_model_per_domain = {}
embeddings_per_domain = {}

# collect all results of all experiments in all given groups
for group_dir in argv:
    print('Starting to load data for group %s' % group_dir)

    domain_name = group_dir.split('/')[-1]
    domain_name = domain_name.replace(group_name_pattern, '')

    domain_names.add(domain_name)

    results_per_domain[domain_name] = {}

    for result_dir in os.listdir(group_dir):
        # skip the old/ directory
        if result_dir == 'old':
            continue

        result_dir = path.join(group_dir, result_dir)
        valid_file = path.join(result_dir, valid_file_name)
        weights_file = path.join(result_dir, weights_file_name)
        params_file = path.join(result_dir, params_file_name)

        if not path.isdir(result_dir):
            print('ERROR: %s is not a directory' % result_dir)
            sys.exit(2)

        if not path.isfile(valid_file):
            print('ERROR: Missing validation metrics in directory %s' % result_dir)
            print('       Expected it at the path %s' % valid_file)
            sys.exit(2)

        if not path.isfile(weights_file):
            print('ERROR: Missing weights file in directory %s' % result_dir)
            print('       Expected it at the path %s' % weights_file)

        if not path.isfile(params_file):
            print('ERROR: Missing params file in directory %s' % result_dir)
            print('       Expected it at the path %s' % params_file)

        params = None
        valid_metrics = None

        with open(params_file, 'r') as f:
            params = json.load(f)

        with open(valid_file, 'r') as f:
            valid_metrics = json.load(f)

        if domain_name not in valid_data_per_domain:
            valid_data_per_domain[domain_name] = params['validation_data_path']

        if domain_name not in params_per_domain:
            params_per_domain[domain_name] = {}

        params_per_domain[domain_name][result_dir] = params
        results_per_domain[domain_name][result_dir] = valid_metrics['all'][0][monitored_metric]

    print('Finished loading data for group %s' % group_dir)

# start to process the collected results to find the best model
# for each domain
for domain in domain_names:
    results = results_per_domain[domain]

    best_result_value = -1.0
    best_result_dir = None

    for res_dir, res_value in results.items():
        if res_value > best_result_value:
            best_result_value = res_value
            best_result_dir = res_dir

    best_result_directory_per_domain[domain] = best_result_dir

# copy the best model to the according directory in models/ for each domain
for domain in domain_names:
    print('Copying the best model for the domain %s' % domain)

    models_path = path.join(models_dir, models_dir_pattern % domain)
    model_weights_path = path.join(models_path, weights_file_name)
    model_config_path = path.join(models_path, model_file_name)

    if path.isdir(models_path):
        print('ERROR: Model directory %s already exists' % models_path)
        print('       Please delete it before regenerating the configs')
        sys.exit(2)
    else:
        os.mkdir(models_path)

    best_res_path = best_result_directory_per_domain[domain]
    weights_path = path.join(best_res_path, weights_file_name)
    model_path = path.join(best_res_path, model_file_name)

    best_model_per_domain[domain] = {
        'model': model_config_path,
        'weights': model_weights_path
    }

    shutil.copyfile(weights_path, model_weights_path)
    shutil.copyfile(model_path, model_config_path)

    print('Copied the best model for the domain %s' % domain)

# start to generate configs
for domain in domain_names:
    print('Starting to generate configs for the domain %s' % domain)

    config_dir = configs_dir_pattern % domain
    best_model = best_model_per_domain[domain]['model']
    best_weights = best_model_per_domain[domain]['weights']

    if path.isdir(config_dir):
        print('ERROR: Config directory %s already exists' % config_dir)
        print('       Please delete it before regenerating the configs')
        sys.exit(2)
    else:
        os.mkdir(config_dir)

    for other_domain in domain_names:
        config_file_name = configs_file_pattern % (domain, other_domain)
        config_file_path = path.join(config_dir, config_file_name)

        best_params = params_per_domain[domain][best_result_directory_per_domain[domain]]

        vocabulary_path = best_params['vocabulary_path']
        embeddings_path = best_params['vocabulary_embeddings']

        new_config = {
            'group_id': group_id_pattern % domain,
            'set_class_weights': True,
            'nb_epoch': 1000,
            'validation_data_path': valid_data_per_domain[other_domain],
            'vocabulary_embeddings': embeddings_path,
            'vocabulary_path': vocabulary_path,
            'model_json_path': best_model_per_domain[domain]['model'],
            'model_weights_path': best_model_per_domain[domain]['weights']
        }

        with open(config_file_path, 'w+') as f:
            json.dump(new_config, f, indent=4, sort_keys=True)

    print('Generated all configs for the domain %s' % domain)

print('Finished preparing all crossdomain_sp_* configs and models')
