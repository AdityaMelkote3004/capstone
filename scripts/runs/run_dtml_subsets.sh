#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
C="--window 10 --beta 1 --hidden 64 --lr 0.0001 --seeds"
{
for s in 0 1 2; do
  echo "python -u scripts/run_dtml.py $C $s --tag S0_all --features 0 1 2 3 4 5 6 7 8 9 10"
  echo "python -u scripts/run_dtml.py $C $s --tag S1_ret --features 3"
  echo "python -u scripts/run_dtml.py $C $s --tag S2_ret_intra --features 0 1 2 3"
  echo "python -u scripts/run_dtml.py $C $s --tag S3_trend --features 5 6 7 8 9 10"
  echo "python -u scripts/run_dtml.py $C $s --tag S4_ret_trend --features 3 5 6 7 8 9 10"
  echo "python -u scripts/run_dtml.py $C $s --tag S5_compact --features 3 5 7 10"
done
} | xargs -P 7 -I{} sh -c '{}'
echo "### SUBSETS COMPLETE"
