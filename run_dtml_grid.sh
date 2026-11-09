#!/bin/bash
# Experiment 11 stage A: DTML's own hyperparameter grid (paper Sec 4.1), DEV ONLY.
# w {10,15} x beta {0.01,0.1,1} x h {64,128} x lr {1e-3,1e-4} = 24 configs x 2 seeds.
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for w in 10 15; do for b in 0.01 0.1 1; do for h in 64 128; do for lr in 0.001 0.0001; do
  echo "--window $w --beta $b --hidden $h --lr $lr --seeds 0 1 --tag gridA"
done; done; done; done | xargs -P 6 -I{} sh -c 'python -u scripts/run_dtml.py {}'
echo "### GRID A COMPLETE"
