#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
A="--two-branch --adv --lambda-aux 1 --tag D2_final --out-prefix exp12b_ --eval-test --save-preds"
{
for s in 0 1 2 3 4 5 6 7 8 9; do echo "python -u scripts/run_dtml_disentangled.py $A --seeds $s --fold main"; done
for f in A B; do for s in 0 1 2 3 4; do echo "python -u scripts/run_dtml_disentangled.py $A --seeds $s --fold $f"; done; done
} | xargs -P 8 -I{} sh -c '{}'
echo "### EXP12B FINAL COMPLETE"
