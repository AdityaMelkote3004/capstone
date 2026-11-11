#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
{
for s in 0 1 2 3 4 5 6 7 8 9; do echo "python -u scripts/run_dtml_disentangled.py --mode struct --adv --lambda-m 1 --seeds $s --tag D_final --fold main --eval-test --save-preds"; done
for f in A B; do for s in 0 1 2 3 4; do echo "python -u scripts/run_dtml_disentangled.py --mode struct --adv --lambda-m 1 --seeds $s --tag D_final --fold $f --eval-test --save-preds"; done; done
} | xargs -P 7 -I{} sh -c '{}'
echo "### EXP12 FINAL COMPLETE"
