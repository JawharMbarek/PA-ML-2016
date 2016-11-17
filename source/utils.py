from datetime import datetime

def generate_test_id(params=None):
    '''This function creates the id for a new test.'''
    name = 'test'

    if params is not None and 'name' in params:
        name = params['name']

    return '%s-%s' % (datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), name)
