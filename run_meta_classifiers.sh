#!/usr/bin/env bash

LIST=(DAI_tweets DIL_reviews HUL_reviews JCR_quotations \
      MPQ_reviews SEM_headlines SemEval_tweets TAC_reviews)

LIST_DOMAINS=(
  "dai,dil,hul,mpq,semeval" 
  "dai,dil,hul,semeval,tac" 
  "dai,dil,mpq,semeval,tac" 
  "dai,hul,mpq,semeval,tac" 
  "dil,hul,mpq,semeval,tac" 
  "dai,dil,hul,mpq,semeval,tac"
)

for domain in "${LIST[@]}"
do
  for used_domains in "${LIST_DOMAINS[@]}"
  do
    echo "Running meta classifier for domain $domain"
    python scripts/meta_classifier_evaluation.py "testdata/${domain}_train.tsv" \
                   "testdata/${domain}_test.tsv" "testdata/${domain}_test.tsv" "${used_domains}" || exit 2
  done
done
