#!/bin/bash
# Revision step 6: federated with a long schedule (pre-registered keep/remove rule).
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1 2 3 4; do
  echo "python -u scripts/run_dtml_federated.py --rounds 150 --patience 30 --seeds $s --tag FEDlong --save-preds --out-prefix rev_"
done | xargs -P 5 -I{} sh -c '{}'
echo "### REV FEDERATED COMPLETE"
