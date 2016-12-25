#!/usr/bin/env bash

LIST=(DAI_tweets DIL_reviews HUL_reviews JCR_quotations MPQ_reviews SEM_headlines SemEval_tweets)

for domain in "${LIST[@]}"
do
  echo "Running meta classifier for domain $domain"
  python scripts/meta_classifier_evaluation.py \
    "testdata/${domain}_train.tsv" "testdata/${domain}_test.tsv"
done
