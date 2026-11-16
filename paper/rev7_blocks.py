"""Review round 7: block-length sensitivity (1 and 10 days) of the Experiment 14
control comparisons (Rank IC, control - MMGT-D2 and control - MMGT). Reported only;
the pre-registered rules were applied with 5-day blocks (rev7_stats.json).
Writes paper/rev7_blocks.json."""
import json, os, sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import rev7_stats as R7  # noqa: E402
import rev6_stats as R6  # noqa: E402


def main():
    days = np.unique(R6.load("R_mmgt")[0]["day"])
    tags = {"MMGT": "R_mmgt", "MMGT-D2": "D2_final", **R7.CONTROLS}
    PD = {n: [R7.per_day(r, days) for r in R6.load(t)] for n, t in tags.items()}
    out = {}
    for c in list(R7.CONTROLS) + ["MMGT-D2"]:
        for ref in ("MMGT-D2", "MMGT"):
            if c == ref:
                continue
            for L in (1, 5, 10):
                r = R7.boot(PD[c], "ic", PD[ref], "ic", L=L)
                out[f"{c} - {ref} | ic | L={L}"] = r
                print(c, "-", ref, "L", L, round(r["point"], 4), [round(x, 4) for x in r["ci"]], round(r["p"], 3))
    json.dump(out, open(os.path.join(H, "rev7_blocks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
