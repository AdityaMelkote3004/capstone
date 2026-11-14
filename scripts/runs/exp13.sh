#!/bin/bash
# Experiment 13 (see PREREGISTRATION.md): MMGT-D2 ablation + market-information probe.
#  - the three MMGT-D2 configurations not selected in Exp 12b, standard test, seeds 0-9
#    (predictions and hidden-state summaries saved)
#  - reruns of D2_final and MMGT with the original settings, states saved only
#    (their original predictions are kept; reproduction is checked against
#    exp12b_final.csv / rev_plus_final.csv)
# Seed 0 of the two reruns was run first as a pilot (out-prefix exp13pilot_).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
D="python -u scripts/run_dtml_disentangled.py --two-branch --out-prefix exp13_ --eval-test --save-states --fold main"
M="python -u scripts/run_dtml_plus.py --window 10 --beta 1 --hidden 64 --lr 0.0001 --extra peer --rank-alpha 1 --tag R_mmgt --fold main --eval-test --save-states --out-prefix exp13_"
{
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "$D --lambda-aux 0.5 --tag D2_la0.5 --save-preds --seeds $s"
  echo "$D --lambda-aux 1 --tag D2_la1 --save-preds --seeds $s"
  echo "$D --adv --lambda-aux 0.5 --tag D2adv_la0.5 --save-preds --seeds $s"
done
for s in 1 2 3 4 5 6 7 8 9; do
  echo "$D --adv --lambda-aux 1 --tag D2_final --seeds $s"
  echo "$M --seeds $s"
done
} | xargs -P 8 -I{} sh -c '{}'
echo "### EXP13 COMPLETE"
