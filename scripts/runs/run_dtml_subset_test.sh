#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "python -u scripts/run_dtml.py --window 10 --beta 1 --hidden 64 --lr 0.0001 --eval-test --seeds $s --tag S4_ret_trend --features 3 5 6 7 8 9 10"
done | xargs -P 7 -I{} sh -c '{}'
echo "### SUBSET TEST COMPLETE"
