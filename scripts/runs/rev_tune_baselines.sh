#!/bin/bash
# Revision step 1: tune baselines on VALIDATION only (no --eval-test).
cd "D:/Moved/Capstone/Implementation/capstone-latest/.claude/worktrees/macro-conditioned"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
{
for C in 0.01 0.1 1 10; do echo "python -u scripts/run_acl18_baselines.py --model lr --C $C --tag T_lr_C$C --out-prefix rev_"; done
for h in 32 64 128; do for lr in 0.001 0.0001; do
  echo "python -u scripts/run_acl18_baselines.py --model mlp --hidden $h --lr $lr --seeds 0 1 --tag T_mlp_h${h}_lr$lr --out-prefix rev_"
done; done
for w in 10 15; do for h in 32 64 128; do for lr in 0.001 0.0001; do
  echo "python -u scripts/run_dtml.py --window $w --hidden $h --lr $lr --ablate all --seeds 0 1 --tag T_lstm_w${w}_h${h}_lr$lr --out-prefix rev_"
  echo "python -u scripts/run_dtml.py --window $w --hidden $h --lr $lr --ablate samc --seeds 0 1 --tag T_alstm_w${w}_h${h}_lr$lr --out-prefix rev_"
done; done; done
} | xargs -P 3 -I{} sh -c '{}'
echo "### REV TUNING COMPLETE"
