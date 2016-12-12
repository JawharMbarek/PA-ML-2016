import sys
import re
import os
import json
import copy

from os import path

'''
{
  "group_id": "crossdomain_sp_allbut_%domain%",
  "set_class_weights": true,
  "vocabulary_embeddings": "embeddings/emb_smiley_tweets_embedding_english_590M.npy",
  "validation_data_path": "testdata/DAI_tweets_test.tsv",
  "vocabulary_path": "vocabularies/vocab_en300M_reduced.pickle",
  "model_json_path": "models/tweets_distant_model_tweet_embeddings/model.json",
  "model_weights_path": "models/tweets_distant_model_tweet_embeddings/weights_80M.h5"
}
'''

tsv_files = sys.argv[1:]

config_dir_pattern = re.compile('crossdomain_sp_allbut_[a-zA-Z0-9]+')
config_path = path.abspath(path.join(path.dirname(__file__), '..', 'configs'))
config_model_pattern = 'crossdomain_allbut_%s_model.json'
config_new_pattern = 'crossdomain_allbut_%s_validate_on_%s.json'

allbut_model_path = 'models/crossdomain_sp_allbut_%s_model/model.json'
allbut_weights_path = 'models/crossdomain_sp_allbut_%s_model/weights_1.h5'

for config_dir in os.listdir(config_path):
  if config_dir_pattern.match(config_dir) and not config_dir.endswith('models'):
    print('Processing the directory %s' % config_dir)

    domain_name = config_dir.split('_')[-1]

    model_params = None
    model_config_file = path.join(config_path, config_dir,
                                  config_model_pattern % domain_name)

    with open(model_config_file, 'r') as f:
      cfg = json.load(f)
      del(cfg['test_data'])
      model_params = cfg

    for val_file in tsv_files:
      val_domain = val_file.split('/')[-1].lower().split('_')[0]

      if val_domain == domain_name:
        continue

      if val_domain == 'union':
        val_domain = val_file.split('/')[-1].lower()
        val_domain = val_domain.split('.')[0]
        val_domain = val_domain.replace('_test', '')

      new_config_path = path.join(config_path, config_dir,
                                  config_new_pattern % (domain_name, val_domain))

      curr_params = copy.deepcopy(model_params)

      curr_params['model_json_path'] = allbut_model_path % domain_name
      curr_params['model_weights_path'] = allbut_weights_path % domain_name
      curr_params['validation_data_path'] = val_file

      with open(new_config_path, 'w+') as f:
        json.dump(curr_params, f, indent=4, sort_keys=True)

  else:
    print('Skipping directory %s' % config_dir)
