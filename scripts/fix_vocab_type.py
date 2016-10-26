#!/usr/bin/env python

import sys
import pickle

argv = sys.argv[1:]
inp = argv[0]
out = argv[1]

v = pickle.load(open(inp, 'rb'))

new_dict = dict()

for key, (idx, _) in v.items():
    new_dict[key] = idx

pickle.dump(dict(new_dict), open(out, 'wb'))

print('DONE')