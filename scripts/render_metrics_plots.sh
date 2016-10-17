#!/bin/bash

declare -a COMBOS=('val_f1_score,val_f1_score_pos_neg' \
                   'val_f1_score_pos,val_f1_score_neg,val_f1_score_neu',
                   'val_acc,acc')

ALL_METRICS=$1/train_metrics_all.json
OPT_METRICS=$1/train_metrics_opt.json

mkdir -p $1/plots/

for c in "${COMBOS[@]}"
do
  python scripts/generate_metrics_plot.py -m $OPT_METRICS -o $c \
                                          -i $1/plots/opt_metrics_$c.png

  echo "Finished plotting the optimal metrics (metrics: $c)"

  for i in {1..10}
  do
    python scripts/generate_metrics_plot.py -m $ALL_METRICS -o $c -n $i \
                                            -i $1/plots/all_metrics_run_${i}_${c}.png
    echo "Finished plotting run $i (metrics: $c)"
  done
done
