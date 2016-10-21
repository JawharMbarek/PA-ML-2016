import os
import flask
import json
import tempfile

from os import path
from subprocess import Popen, PIPE

app = flask.Flask(__name__)

dirname = path.dirname(__file__)
index_path = path.join(dirname, 'index.html')
results_path = path.join(dirname, os.pardir, 'results')
gen_plot_path = path.join(dirname, os.pardir, 'scripts', 'generate_metrics_plot.py')
gen_pp_plot_path = path.join(dirname, os.pardir, 'scripts', 'generate_per_percentage_plot.py')

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

@app.route('/generate_plot')
def generate_plot():
    with tempfile.NamedTemporaryFile(suffix='.png') as tf:
        tf_name = tf.name
        plot_type = flask.request.args.get('plot_type')
        only_metrics = flask.request.args.get('metrics')
        full_name = flask.request.args.get('full_name')
        metrics_file_path = os.path.join(results_path, full_name, 'train_metrics_opt.json')

        print('Using %s as a tempfile' % tf_name)
        print('For the metrics file %s' % metrics_file_path)

        if plot_type == 'standard':
            plot_proc = Popen([
                'python',
                gen_plot_path,
                '-i', tf_name,
                '-o', only_metrics,
                '-m', metrics_file_path
            ], stdout=PIPE, stderr=PIPE)

            plot_proc.communicate()
            plot_err = plot_proc.returncode

            print('generate_metrics_plot.py return the error code %s' % plot_err)

            if plot_err == 0:
                return flask.send_file(tf_name, mimetype='image/png')
            else:
                return '500 error', 500
        elif plot_type == 'per-percentage':
            groupid = path.split(full_name)[0]

            plot_proc = Popen([
                'python',
                gen_pp_plot_path,
                '-r', path.join(results_path, groupid),
                '-i', tf_name
            ], stdout=PIPE, stderr=PIPE, shell=True)

            output = plot_proc.communicate()
            plot_err = plot_proc.returncode

            print('generate_per_percentage_plot.py return the error code %s' % plot_err)
            print('output: %s' % str(output))

            if plot_err == 0:
                return flask.send_file(tf_name, mimetype='image/png')
            else:
                return '500 error', 500


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