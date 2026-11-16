#!/bin/bash
# Experiment 14 (see PREREGISTRATION.md): architectural controls requested in review round 7.
#  C1 D2_unc     two-branch without the zero-sum constraint
#  C2 D2_noatt   two-branch, selection branch without stock-to-stock attention
#  C3 MMGT_wide  single-branch MMGT with hidden 92 (parameter-matched to MMGT-D2)
#  D2_final seed 0 rerun with the modified code (reproduction check)
# Each job writes its own CSV under exp14/ so parallel jobs never append to one file.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p results/experiment11_dtml/exp14
D="python -u scripts/run_dtml_disentangled.py --two-branch --adv --lambda-aux 1 --eval-test --save-preds --fold main"
M="python -u scripts/run_dtml_plus.py --window 10 --beta 1 --hidden 92 --lr 0.0001 --extra peer --rank-alpha 1 --tag MMGT_wide --fold main --eval-test --save-preds"
{
echo "$D --tag D2_final_repro --save-states --out-prefix exp14/D2_final_repro_s0_ --seeds 0"
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "$D --unconstrained --tag D2_unc --save-states --out-prefix exp14/D2_unc_s${s}_ --seeds $s"
  echo "$D --no-select-attention --tag D2_noatt --save-states --out-prefix exp14/D2_noatt_s${s}_ --seeds $s"
  echo "$M --out-prefix exp14/MMGT_wide_s${s}_ --seeds $s"
done
} | xargs -P 7 -I{} sh -c '{}'
echo "### EXP14 COMPLETE"
