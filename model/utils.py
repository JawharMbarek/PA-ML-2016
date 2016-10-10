import numpy
import time

from datetime import datetime

def generate_test_id(params=None):
    '''This function creates the id for a new test.'''
    now = datetime.now()
    name = 'test'

    if params is not None and 'name' in params:
        name = params['name']

    return '%d-%d-%d-%d-%s' % (now.year, now.month, now.day, time.time(), name)
