import os
import sys
import csv
import numpy as np

from os import path
from collections import defaultdict

argv = sys.argv[1:]

if len(argv) < 1:
  print('ERROR: Missing mandatory arguments!')
  print('       (./combine_model_predictions_for_meta_classifier.py <predictions-directory> [<prediction-values=probas|onehot>')
  sys.exit(2)

pred_dir = argv[0]
pred_values = 'probas'
domains = set()

if len(argv) > 1:
  pred_values = argv[1]

  if pred_values != 'probas' and pred_values != 'onehot':
    print('ERROR: prediction-values has to be set to either "probas" or "onehot"!')
    sys.exit(2)

num_lines_per_dataset = {}
files_to_combine = defaultdict(lambda: [])
prediction_files = list(sorted(filter(lambda x: not x.startswith('combined_'), os.listdir(pred_dir))))

for file_name in prediction_files:
  file_path = path.join(pred_dir, file_name)
  num_lines = sum([1 for _ in open(file_path, 'r')])
  file_name_parts = file_name.split('_on_')
  domain_name, dataset_name = file_name_parts

  files_to_combine[dataset_name].append(file_name)

  if dataset_name not in num_lines_per_dataset:
    num_lines_per_dataset[dataset_name] = num_lines

  if num_lines_per_dataset[dataset_name] != num_lines:
    print('Inconsistent file size for domain %s! (%d != %d)' % (
      dataset_name, num_lines_per_dataset[dataset_name], num_lines))

files_contents = {}
dataset_keys = list(sorted(files_to_combine.keys()))

def str_to_float_list(string):
  # Ugly...
  elements = string[2:-1].split(' ')
  elements = filter(lambda x: len(x) > 0, elements)
  return list(map(float, elements))

for files_to_load in files_to_combine.values():
  for f_name in files_to_load:
    f_path = path.join(pred_dir, f_name)

    with open(f_path, 'r') as f:
      csv_reader = csv.reader(f)
      next(csv_reader) # skip headers
      files_contents[f_name] = [tuple(l) for l in csv_reader]

for ds_key in dataset_keys:
  out_file_name = 'combined_%s' % ds_key
  out_file_path = path.join(pred_dir, out_file_name)

  combined_predictions = []

  for inc_file in files_to_combine[ds_key]:
    for i, entry in enumerate(files_contents[inc_file]):
      if len(combined_predictions) == i:
        combined_predictions.append([entry[0],
                                     str_to_float_list(entry[1]),
                                     str_to_float_list(entry[2])])
      else:
        combined_predictions[i][2] += str_to_float_list(entry[2])

  with open(out_file_path, 'w+') as out_f:
    csv_writer = csv.writer(out_f)
    csv_writer.writerow(['Text', 'Sentiment', 'Predictions'])

    for txt, sent, pred in combined_predictions:
      csv_writer.writerow([txt.strip('\n'), str(sent), str(pred)])

  print('Finished combining all predictions for dataset %s!' % ds_key)
  