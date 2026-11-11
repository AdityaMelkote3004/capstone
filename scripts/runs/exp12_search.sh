#!/bin/bash
# Experiment 12 validation grid (pre-registered): {struct, beta, struct+adv} x lambda_m {0.5, 1}, seeds 0 1.
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
{
for lm in 0.5 1; do for s in 0 1; do
  echo "python -u scripts/run_dtml_disentangled.py --mode struct --lambda-m $lm --seeds $s --tag D_struct_lm$lm"
  echo "python -u scripts/run_dtml_disentangled.py --mode beta --lambda-m $lm --seeds $s --tag D_beta_lm$lm"
  echo "python -u scripts/run_dtml_disentangled.py --mode struct --adv --lambda-m $lm --seeds $s --tag D_adv_lm$lm"
done; done
} | xargs -P 6 -I{} sh -c '{}'
echo "### EXP12 SEARCH COMPLETE"
