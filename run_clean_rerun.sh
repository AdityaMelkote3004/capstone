#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
C="python -u scripts/run_dtml_plus.py --window 10 --beta 1 --hidden 64 --lr 0.0001 --extra peer --eval-test"
{
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "$C --seeds $s --tag F1c"
  echo "$C --seeds $s --graph bias --relations sector correlation comention residual --tag G2c"
  echo "$C --seeds $s --rank-alpha 1 --tag F2c"
done
for s in 0 1 2 3 4; do echo "$C --seeds $s --rank-alpha 1 --ablate mc --tag F3c"; done
} | xargs -P 7 -I{} sh -c '{}'
echo "### CLEAN MAIN COMPLETE"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for s in 0 1 2 3 4; do echo "python -u scripts/run_dtml_federated.py --seeds $s --tag FEDc"; done | xargs -P 5 -I{} sh -c '{}'
echo "### CLEAN FED COMPLETE"
