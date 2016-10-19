from time import gmtime, strftime

def generate_test_id(params=None):
    '''This function creates the id for a new test.'''
    now = gmtime()
    name = 'test'

    if params is not None and 'name' in params:
        name = params['name']

    return '%s-%s' % (strftime('%Y-%m-%d-%H-%M-%S', now), name)
