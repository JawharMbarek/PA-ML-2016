#!/usr/bin/env bash

LIST=(DAI_tweets DIL_reviews HUL_reviews JCR_quotations \
      MPQ_reviews SEM_headlines SemEval_tweets TAC_reviews)

LIST_DOMAINS=(
  # SemEval13
  "dai,dil,hul,mpq,semeval,tac,jcr"
  "dai,dil,hul,mpq,semeval,tac,sem"
  "dai,dil,hul,mpq,semeval,jcr,sem"
  "dai,dil,hul,mpq,tac,jcr,sem"
  "dai,dil,hul,semeval,tac,jcr,sem"
  "dai,dil,mpq,semeval,tac,jcr,sem"
  "dai,hul,mpq,semeval,tac,jcr,sem"
  "dil,hul,mpq,semeval,tac,jcr,sem"
  "dai,dil,hul,mpq,semeval,tac,jcr,sem"
  # SemEval16
  "dai,dil,hul,mpq,semeval16,tac,jcr"
  "dai,dil,hul,mpq,semeval16,tac,sem"
  "dai,dil,hul,mpq,semeval16,jcr,sem"
  "dai,dil,hul,mpq,tac,jcr,sem"
  "dai,dil,hul,semeval16,tac,jcr,sem"
  "dai,dil,mpq,semeval16,tac,jcr,sem"
  "dai,hul,mpq,semeval16,tac,jcr,sem"
  "dil,hul,mpq,semeval16,tac,jcr,sem"
  "dai,dil,hul,mpq,semeval16,tac,jcr,sem"
  # SemEval17
  "dai,dil,hul,mpq,semeval17,tac,jcr"
  "dai,dil,hul,mpq,semeval17,tac,sem"
  "dai,dil,hul,mpq,semeval17,jcr,sem"
  "dai,dil,hul,mpq,tac,jcr,sem"
  "dai,dil,hul,semeval17,tac,jcr,sem"
  "dai,dil,mpq,semeval17,tac,jcr,sem"
  "dai,hul,mpq,semeval17,tac,jcr,sem"
  "dil,hul,mpq,semeval17,tac,jcr,sem"
  "dai,dil,hul,mpq,semeval17,tac,jcr,sem"
)

for domain in "${LIST[@]}"
do
  for used_domains in "${LIST_DOMAINS[@]}"
  do
    echo "Running meta classifier for domain $domain using the following domains: $used_domains"
    python scripts/meta_classifier_evaluation.py "testdata/${domain}_train.tsv" \
                   "testdata/${domain}_test.tsv" "testdata/${domain}_test.tsv" "${used_domains}" || exit 2
  done
done
