"""Classical volatility baselines for Experiment 9.

Experiment 9 reports MCC +0.1433 for predicting whether tomorrow's absolute move
exceeds the training median. That number means nothing until it is compared
against what volatility forecasting has done for thirty years. If GARCH matches
or beats it, the neural result is a reimplementation of a solved problem; if the
LSTM adds something, that is worth saying -- but only with the baseline in hand.

Three baselines, cheapest first:

  realised_k   std of the previous k returns (k = 5, 10, 20)
  ewma         RiskMetrics exponentially-weighted vol, lambda = 0.94
  garch11      GARCH(1,1), refit per ticker on train returns, then filtered
               forward through dev/test

All use returns STRICTLY BEFORE the target date. The GARCH parameters are
estimated on the training period only and never refit on test, so no test
information reaches the model.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

from src.utils.metrics import within_day_metrics

warnings.filterwarnings("ignore")

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
OUT = "results/experiment9_volatility/garch_baseline.json"
TEXT_CACHE = "data/processed/experiment8_text_matrix.pt"
TRAIN_END = "2015-08-01"          # protocol-A train boundary


def build_vol_features() -> pd.DataFrame:
    """Per (ticker, date), volatility estimates using only strictly-prior returns."""
    hist = pd.read_parquet(FINAL, columns=["Date", "Ticker", "Return"])
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    grouped = hist.groupby("Ticker")["Return"]
    for k in (5, 10, 20):
        # shift(1) is what makes this strictly prior: the window ends at t-1.
        hist[f"realised_{k}"] = grouped.transform(
            lambda s, k=k: s.shift(1).rolling(k, min_periods=2).std())
    hist["ewma"] = grouped.transform(
        lambda s: np.sqrt(s.shift(1).pow(2).ewm(alpha=1 - 0.94).mean()))
    return hist


def add_garch(hist: pd.DataFrame) -> pd.DataFrame:
    """One-step-ahead GARCH(1,1) conditional vol, parameters fit on train only."""
    from arch import arch_model

    out = np.full(len(hist), np.nan)
    for ticker, block in hist.groupby("Ticker"):
        returns = block["Return"].to_numpy(dtype=float) * 100.0     # arch likes %
        train_mask = (block["Date"] < TRAIN_END).to_numpy()
        if train_mask.sum() < 100 or len(returns) < 120:
            continue
        fitted = arch_model(returns[train_mask], p=1, q=1, vol="Garch",
                            dist="normal").fit(disp="off", show_warning=False)
        mu = float(fitted.params["mu"])
        omega = float(fitted.params["omega"])
        alpha = float(fitted.params["alpha[1]"])
        beta = float(fitted.params["beta[1]"])

        # Run the GARCH(1,1) recursion forward over the FULL series using
        # train-estimated parameters. arch's own forecast() only spans the
        # window it was fitted on, so doing it explicitly is what lets the
        # variance propagate into dev/test without ever re-estimating there.
        #   sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
        # sigma2_t depends only on information through t-1, so it is a genuine
        # one-step-ahead forecast at every point.
        eps = returns - mu
        sigma2 = np.empty(len(returns))
        sigma2[0] = omega / max(1.0 - alpha - beta, 1e-6)   # unconditional var
        for i in range(1, len(returns)):
            sigma2[i] = omega + alpha * eps[i - 1] ** 2 + beta * sigma2[i - 1]
        out[block.index.to_numpy()] = np.sqrt(sigma2)

    hist["garch11"] = out
    return hist


def evaluate(name: str, train_x, train_y, test_x, test_y, test_days) -> dict:
    model = LogisticRegression(max_iter=2000).fit(train_x, train_y)
    prob = model.predict_proba(test_x)[:, 1]
    pred = (prob >= 0.5).astype(int)
    result = {
        "baseline": name,
        "mcc": float(matthews_corrcoef(test_y, pred)),
        "accuracy": float(accuracy_score(test_y, pred)),
        "auc": float(roc_auc_score(test_y, prob)),
        "n_samples": int(len(test_y)),
    }
    result.update({f"within_day_{k}": v
                   for k, v in within_day_metrics(test_y, prob, test_days).items()})
    return result


def main() -> None:
    hist = add_garch(build_vol_features())

    df = pd.read_parquet(EXACT)
    df["Date"] = pd.to_datetime(df["Target_Date"])
    df = df.merge(hist.drop(columns=["Return"]), on=["Ticker", "Date"], how="left")

    magnitude = df["Main_MV_Percent"].abs()
    threshold = magnitude[df["Split"] == "train"].median()
    df["vol_target"] = (magnitude > threshold).astype(int)
    df["day"] = df["Date"].dt.strftime("%Y-%m-%d")

    # The strongest test of "text adds nothing": bolt it onto the BEST baseline,
    # not onto our own models (which lose to EWMA). If text cannot improve EWMA,
    # the negative result holds against the best thing available, not just
    # against what we built.
    text_matrix = torch.load(TEXT_CACHE, weights_only=False)["matrix"].numpy()
    assert len(text_matrix) == len(df), "text matrix must align with the EXACT parquet order"
    is_train = (df["Split"] == "train").to_numpy()
    centre = text_matrix[is_train].mean(axis=0)
    _, _, vt = np.linalg.svd(text_matrix[is_train] - centre, full_matrices=False)
    projected = (text_matrix - centre) @ vt[:8].T          # PCA fit on train only
    for i in range(8):
        df[f"pca{i}"] = projected[:, i]
    df["tweet_total"] = np.log1p(
        df[[f"Day{i}_TweetCount" for i in range(1, 5)]].to_numpy().clip(0).sum(axis=1))
    PCA_COLS = [f"pca{i}" for i in range(8)]

    feature_sets = {
        "realised_5": ["realised_5"],
        "realised_10": ["realised_10"],
        "realised_20": ["realised_20"],
        "ewma_riskmetrics": ["ewma"],
        "garch11": ["garch11"],
        "all_classical": ["realised_5", "realised_10", "realised_20", "ewma", "garch11"],
        "ewma + tweet counts": ["ewma", "tweet_total"],
        "ewma + PCA-8 text": ["ewma"] + PCA_COLS,
        "ewma + counts + text": ["ewma", "tweet_total"] + PCA_COLS,
    }

    rows = []
    for name, cols in feature_sets.items():
        usable = df.dropna(subset=cols)
        train = usable[usable.Split == "train"]
        test = usable[usable.Split == "test"]
        if len(train) < 500 or len(test) < 500:
            print(f"  {name:<18} SKIPPED (train {len(train)}, test {len(test)})")
            continue
        result = evaluate(name, train[cols], train.vol_target,
                          test[cols], test.vol_target, test["day"].to_numpy())
        rows.append(result)
        print(f"  {name:<18} MCC={result['mcc']:+.4f}  "
              f"within-day={result['within_day_within_day_mcc']:+.4f}  "
              f"acc={result['accuracy']:.4f}  AUC={result['auc']:.4f}  "
              f"n={result['n_samples']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_json(OUT, orient="records", indent=2)
    print(f"\nNeural (Exp 9, price-only LSTM): MCC +0.1433, within-day +0.1322, "
          f"acc 0.5848, AUC 0.6002, n=3720")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
