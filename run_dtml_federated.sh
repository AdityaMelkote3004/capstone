#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for s in 0 1 2 3 4; do echo "python -u scripts/run_dtml_federated.py --seeds $s"; done | xargs -P 5 -I{} sh -c '{}'
echo "### FEDERATED COMPLETE"
