#!/usr/bin/env python

import os
import sys
import shutil

from os import path

if len(sys.argv) < 5:
    print('ERROR: python scripts/copy_experiment_group.py <source-group> <target-group>')
    print('                                               <old-version> <new-version>')
    sys.exit(2)

argv = sys.argv[1:]

configs_path = path.join(path.dirname(__file__), '..', 'configs')

source_group = argv[0]
target_group = argv[1]

old_version = int(argv[2])
new_version = int(argv[3])

source_path = path.join(configs_path, source_group)
target_path = path.join(configs_path, target_group)

for dir in os.listdir(configs_path):
    if dir == source_group:
        print('Source group "%s" found!' % source_group)

        if path.exists(target_path):
            print('ERROR: Target group %s already exists at %s' % (
                  target_group, path.relpath(target_path)))
            sys.exit(2)

        shutil.copytree(source_path, target_path)

        for f in os.listdir(target_path):
            old_version_str = 'v%d' % old_version
            new_version_str = 'v%d' % new_version

            new_file_name = f.replace(old_version_str, new_version_str)

            old_full_path = path.join(target_path, f)
            new_full_path = path.join(target_path, new_file_name)

            old_content = ''
            new_content = ''

            if old_version_str in f:
                os.rename(old_full_path, new_full_path)

            if f.endswith('.json'):
                with open(new_full_path, 'r') as f:
                    old_content = f.read()
                    new_content = old_content.replace(old_version_str, new_version_str)

                with open(new_full_path, 'w+') as f:
                    f.write(new_content)

print('Successfully copied experiment %s to %s' % (source_group, target_group))
