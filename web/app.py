import os
import flask
import json

from os import path

app = flask.Flask(__name__)

dirname = path.dirname(__file__)
index_path = path.join(dirname, 'index.html')
results_path = path.join(dirname, os.pardir, 'results')

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/results')
def results():
    '''Returns all existing groupids as a JSON array.'''
    return flask.jsonify(get_group_ids())

@app.route('/results/<groupid>')
def results_groupid(groupid):
    '''Returns the names of all experiments with the given
       group id as a JSON array.'''
    return flask.jsonify(get_exp_names(groupid))

@app.route('/results/<groupid>/<exp>')
def results_groupid_exp(groupid, exp):
    '''Returns a JSON object containing the content
       of all json files in the results directory.'''
    metrics_files = {}
    exp_path = path.join(results_path, groupid, exp)

    for f in os.listdir(exp_path):
        file_path = path.join(exp_path, f)

        if f.endswith('.json') and path.isfile(file_path):
            with open(file_path, 'r') as fd:
                metrics_files[f] = json.load(fd)

    return flask.jsonify(metrics_files)

def get_exp_names(groupid):
    groupid = groupid.replace('/', '')
    groupid_path = path.join(results_path, groupid)
    groupid_exps = list(os.listdir(groupid_path))

    return filter_dirs(groupid_exps, groupid_path)

def get_group_ids():
    return filter_dirs(list(os.listdir(results_path)), results_path)

def filter_dirs(dirs, base=None):
    return [x for x in dirs if not x.startswith('.') and
                               path.isdir(path.join(base, x))]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')