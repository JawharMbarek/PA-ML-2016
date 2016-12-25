#!/usr/bin/env bash

LIST=(DAI_tweets DIL_reviews HUL_reviews JCR_quotations \
      MPQ_reviews SEM_headlines SemEval_tweets TAC_reviews)

for domain in "${LIST[@]}"
do
  echo "Running meta classifier for domain $domain"
  python scripts/meta_classifier_evaluation.py \
    "testdata/${domain}_valid.tsv" "testdata/${domain}_test.tsv"
done
