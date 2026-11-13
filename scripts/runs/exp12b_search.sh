#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
{
for la in 0.5 1; do for s in 0 1; do
  echo "python -u scripts/run_dtml_disentangled.py --two-branch --lambda-aux $la --seeds $s --tag D2_la$la --out-prefix exp12b_"
  echo "python -u scripts/run_dtml_disentangled.py --two-branch --adv --lambda-aux $la --seeds $s --tag D2adv_la$la --out-prefix exp12b_"
done; done
} | xargs -P 8 -I{} sh -c '{}'
echo "### EXP12B SEARCH COMPLETE"
