#!/bin/bash
# Stage C, DEV ONLY: one change per rung, 2 seeds each, select on within-day dev MCC.
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R="python -u scripts/run_dtml_plus.py --seeds 0 1 --select within_day_mcc"
cat <<LIST | xargs -P 2 -I{} sh -c "$R {}"
--tag C0_faithful
--tag C1_base --extra base
--tag C2_peer --extra peer
--tag C3_rank --rank-alpha 1
--tag C4_peer_rank --extra peer --rank-alpha 1
--tag C5_peer_rank_nomc --extra peer --rank-alpha 1 --ablate mc
LIST
echo "### STAGE C COMPLETE"
