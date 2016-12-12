#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo 'Wrong number of parameters: ./run_in_background.sh <configs-pattern> <random-seed>'
  exit 2
fi

time ls --color=never $1 | NP_RAND_SEED=$2 xargs ./run.sh

