#!/bin/bash
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
R="scripts/run_experiment10_multimodal.py"
# Matched 5-seed baseline for the direction peer comparison. Stage 4 ran
# direction+peer at 5 seeds but direction plain existed only at seed 0, so the
# contrast had no matched control.
echo "### GAP-FILL -- direction plain, 5 seeds"
python -u $R --full gat --targets direction --variants plain --seeds 0 1 2 3 4
echo "### GAP-FILL COMPLETE"
