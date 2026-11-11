#!/bin/bash
# Revision step 2: final test runs with tuned baselines, predictions saved, plus walk-forward folds.
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="--eval-test --save-preds --out-prefix rev_"
DT="--window 10 --beta 1 --hidden 64 --lr 0.0001"
{
for f in main A B; do
  if [ $f = main ]; then SEEDS="0 1 2 3 4 5 6 7 8 9"; else SEEDS="0 1 2 3 4"; fi
  for s in $SEEDS; do
    echo "python -u scripts/run_dtml_plus.py $DT --extra peer --rank-alpha 1 --seeds $s --tag R_mmgt --fold $f $P"
    echo "python -u scripts/run_dtml.py $DT --seeds $s --tag R_base --fold $f $P"
    echo "python -u scripts/run_dtml.py --window 10 --hidden 64 --lr 0.001 --ablate all --seeds $s --tag R_lstm --fold $f $P"
    echo "python -u scripts/run_acl18_baselines.py --model mlp --hidden 64 --lr 0.001 --seeds $s --tag R_mlp --fold $f $P"
    if [ $f = main ]; then
      echo "python -u scripts/run_dtml_plus.py $DT --extra peer --seeds $s --tag R_profile --fold $f $P"
      echo "python -u scripts/run_dtml.py --window 15 --hidden 128 --lr 0.0001 --ablate samc --seeds $s --tag R_alstm --fold $f $P"
    fi
  done
  echo "python -u scripts/run_acl18_baselines.py --model lr --C 0.01 --seeds 0 --tag R_lr --fold $f $P"
done
} | xargs -P 5 -I{} sh -c '{}'
echo "### REV FINAL COMPLETE"
