import sys
import os
import csv
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: Missing mandatory arguments!')
    print('       (./random_forrest_classifier.py <train-csv> <test-csv>')
    sys.exit(2)

train_csv_path = argv[0]
test_csv_path = argv[1]

def parse_np_array(string):
  return np.array(list(map(float, string[1:-1].split(', '))))

def read_dataset(csv_path):
  X = []
  Y = []

  with open(csv_path, 'r') as csv_f:
    csv_reader = csv.reader(csv_f)
    next(csv_reader) # skip headers

    for txt, sent, inp in csv_reader:
      X.append(parse_np_array(inp))
      Y.append(np.argmax(parse_np_array(sent)))

  return X, Y

train_X, train_Y = read_dataset(train_csv_path)
test_X, test_Y = read_dataset(test_csv_path)

classifier = RandomForestClassifier(n_estimators=300)
classifier = classifier.fit(train_X, train_Y)

test_Y_pred = classifier.predict(test_X)
f1_score = sklearn.metrics.f1_score(test_Y, test_Y_pred, labels=[0, 2], average='macro')

print('RandomForestClassifier on %s resulted in f1_pos_neg=%.3f' % (test_csv_path, f1_score))