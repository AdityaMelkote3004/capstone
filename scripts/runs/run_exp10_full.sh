#!/bin/bash
# Sequential, no wait-loops. Each stage answers one question.
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
R="scripts/run_experiment10_multimodal.py"

echo "### STAGE 1 -- baseline at 5 seeds (graph vs no-graph, plain message passing)"
python -u $R --full gat --targets volatility --variants plain --seeds 0 1 2 3 4

echo "### STAGE 2 -- THE KEY TEST: graph INFORMATION as explicit peer features"
echo "### If no-graph+peer beats no-graph, the edges carry signal that GNN averaging missed."
python -u $R --full gat --targets volatility --variants plain --peer --seeds 0 1 2 3 4

echo "### STAGE 3 -- do the two architectural fixes rescue message passing?"
python -u $R --full gat --targets volatility --variants contrast xsect --seeds 0 1 2 3 4

echo "### STAGE 4 -- repeat the key test on direction"
python -u $R --full gat --targets direction --variants plain --peer --seeds 0 1 2 3 4
echo "### ALL STAGES COMPLETE"
