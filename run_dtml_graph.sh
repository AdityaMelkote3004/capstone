#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
C="python -u scripts/run_dtml_plus.py --window 10 --beta 1 --hidden 64 --lr 0.0001 --extra peer"
{
for s in 0 1 2; do
  echo "$C --seeds $s --graph bias --relations sector correlation comention --tag G1_bias3"
  echo "$C --seeds $s --graph bias --relations sector correlation comention residual --tag G2_bias4"
  echo "$C --seeds $s --graph mask --relations sector correlation comention residual --tag G3_mask4"
done
} | xargs -P 7 -I{} sh -c '{}'
echo "### GRAPH SCREEN COMPLETE"
