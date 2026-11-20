"""Build the compact data file the demo app reads (demo/demo_data.parquet).

It gathers the saved test predictions of the paper's runs (standard test period,
Oct-Dec 2015, every stock with a full 10-day input window, labelled or not) into
one table, so the demo runs anywhere without the 178 prediction files.

Run from the repository root:  python demo/build_demo_data.py
Then check that the printed scores match Table I of the paper.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PREDS = os.path.join(ROOT, "results", "experiment11_dtml", "preds")
OUT = os.path.join(ROOT, "demo", "demo_data.parquet")

# display name -> saved-prediction tag (standard split)
MODELS = {
    "Logistic regression": "R_lr",
    "MLP": "R_mlp",
    "LSTM": "R_lstm",
    "ALSTM": "R_alstm",
    "MMGT-base": "R_base",
    "MMGT": "R_mmgt",
    "MMGT, federated": "FEDlong",
    "MMGT-D": "D_final",
    "MMGT-D2": "D2_final",
    "Control: no zero-sum constraint": "D2_unc",
    "Control: no cross-stock attention": "D2_noatt",
    "Control: wider single MMGT": "MMGT_wide",
}


def main():
    tickers = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "data", "acl18", "ourpped"))
                     if f.endswith(".csv"))
    sectors = pd.read_csv(os.path.join(ROOT, "dataset", "final", "sector_mapping.csv"))
    sector_of = dict(zip(sectors.Ticker, sectors.Sector.str.replace("_", " ")))
    frames = []
    for name, tag in MODELS.items():
        files = sorted(glob.glob(os.path.join(PREDS, f"{tag}_main_s*.npz")),
                       key=lambda s: int(re.search(r"_s(\d+)\.npz$", s).group(1)))
        if not files:
            raise SystemExit(f"missing predictions for {name} ({tag})")
        for f in files:
            z = np.load(f)
            seed = int(re.search(r"_s(\d+)\.npz$", f).group(1))
            df = pd.DataFrame({
                "model": name, "seed": seed,
                "date": z["dates"][z["day"]],
                "stock": z["stock"].astype(np.int16),
                "prob": z["prob"].astype(np.float32),
                "label": z["label"].astype(np.int8),          # 1 up, 0 down, -1 neutral band
                "ret": z["ret"].astype(np.float32),           # close-to-close return, percent
            })
            df["m"] = z["m"].astype(np.float32) if "m" in z.files else np.nan
            df["e"] = z["e"].astype(np.float32) if "e" in z.files else np.nan
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["ticker"] = [tickers[i] for i in data["stock"]]
    data["sector"] = data["ticker"].map(sector_of).fillna("Unknown")
    data = data.drop(columns="stock")
    for c in ("model", "date", "ticker", "sector"):
        data[c] = data[c].astype("category")
    data.to_parquet(OUT, index=False, compression="zstd")
    print(f"wrote {OUT}: {len(data):,} rows, {os.path.getsize(OUT) / 1e6:.1f} MB")

    from demo.metrics import scoreboard   # check against Table I
    board = scoreboard(data)
    board.to_csv(os.path.join(ROOT, "demo", "scoreboard.csv"))
    print(board.round(3).to_string())


if __name__ == "__main__":
    main()
