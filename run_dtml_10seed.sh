#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
C="--window 10 --beta 1 --hidden 64 --lr 0.0001 --eval-test"
{
for s in 5 6 7 8 9; do
  echo "python -u scripts/run_dtml.py $C --seeds $s --tag F0_dtml"
  echo "python -u scripts/run_dtml_plus.py $C --seeds $s --extra peer --tag F1_peer"
  echo "python -u scripts/run_dtml_plus.py $C --seeds $s --extra peer --rank-alpha 1 --tag F2_ours"
done
} | xargs -P 7 -I{} sh -c '{}'
echo "### 10SEED COMPLETE"
