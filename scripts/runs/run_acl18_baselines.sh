#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
until grep -q "GRAPH SCREEN COMPLETE" results/experiment11_dtml/graph_screen.log; do sleep 20; done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
C="--window 10 --beta 1 --hidden 64 --lr 0.0001 --eval-test"
{
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "python -u scripts/run_acl18_baselines.py --model mlp --seeds $s"
  echo "python -u scripts/run_dtml.py $C --seeds $s --ablate all --tag B_lstm"
  echo "python -u scripts/run_dtml.py $C --seeds $s --ablate samc --tag B_alstm"
done
} | xargs -P 7 -I{} sh -c '{}'
echo "### BASELINES COMPLETE"
