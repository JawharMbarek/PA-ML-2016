#!/usr/bin/env python

import os
import sys
import re
import shutil 

from os import path

argv = sys.argv[1:]

dir_pattern = re.compile('\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-[a-zA-Z0-9-_]+')

if len(argv) < 1:
    print('ERROR: Missing mandatory argument')
    print('       python scripts/retire_group_results.py <group-dirs>')
    sys.exit(2)

for group_dir in argv:
    group_old_dir = path.join(group_dir, 'old')

    if not path.isdir(group_dir):
        print('ERROR: The given path %s is not a directory' % group_dir)
        sys.exit(2)

    if not path.isdir(group_old_dir):
        print('The old directory is missing in %s, creating it' % group_dir)
        os.mkdir(group_old_dir)

    result_dirs = os.listdir(group_dir)
    result_dirs = filter(lambda x: dir_pattern.match(x), result_dirs)
    result_dirs = list(map(lambda x: path.join(group_dir, x), result_dirs))

    old_dirs = os.listdir(group_old_dir)
    old_dir_nrs = list(map(lambda x: int(x), old_dirs))
    old_dir_new_nr = -1

    if len(old_dir_nrs) > 0:
        old_dir_new_nr = max(old_dir_nrs) + 1
    else:
        old_dir_new_nr = 1

    old_dir_new = path.join(group_old_dir, str(old_dir_new_nr))

    if path.isdir(old_dir_new):
        print('ERROR: Somehow the old directory %s already exists' % old_dir_new)
        print('       This is kinda strange, exiting...')
        sys.exit(2)
    else:
        os.mkdir(old_dir_new)

    for res_dir in result_dirs:
        dir_name = res_dir.split('/')[-1]
        new_dir_name = path.join(old_dir_new, dir_name)

        shutil.move(res_dir, new_dir_name)

        print('Moved result directory %s to %s' % (dir_name, new_dir_name))
