"""Per-ticker and per-sector breakdown for all 12 baseline experiments."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.data.stocknet_dataset import build_datasets, FEATURE_SETS
from src.models.baselines import LSTMBaseline, MLPBaseline
from src.utils.seed import set_seed

PARQUET = 'dataset/stocknet_final_modeling_set.parquet'
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED    = 42
MIN_SAMPLES = 5   # skip tickers/sectors with too few samples


def metrics_for_group(y_true, y_pred):
    if len(y_true) < MIN_SAMPLES:
        return None
    return {
        'accuracy':  round(float(accuracy_score(y_true, y_pred)), 4),
        'f1':        round(float(f1_score(y_true, y_pred, average='binary', zero_division=0)), 4),
        'mcc':       round(float(matthews_corrcoef(y_true, y_pred)), 4),
        'n_samples': int(len(y_true)),
        'pct_up':    round(float(np.mean(y_true)), 4),
    }


def predict_torch(model, dataset):
    """Return (preds, targets) arrays plus per-sample ticker/sector metadata."""
    model.eval()
    all_preds, all_targets = [], []
    tickers, sectors = [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            window  = batch['window'].to(DEVICE)
            flat    = batch['flat'].to(DEVICE)
            target  = batch['target']
            logits  = model(window=window, flat=flat)
            preds   = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.numpy())
            bsz = len(preds)
            for i in range(bsz):
                meta = dataset.get_metadata(idx)
                tickers.append(meta['ticker'])
                idx += 1
    return np.array(all_preds), np.array(all_targets), tickers


def predict_lr(train_ds, test_ds):
    X_tr, y_tr = train_ds.to_numpy(use_window=False)
    X_te, y_te = test_ds.to_numpy(use_window=False)
    lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    lr.fit(X_tr, y_tr)
    preds = lr.predict(X_te)
    tickers = [test_ds.samples[i]['ticker'] for i in range(len(test_ds))]
    return preds, y_te, tickers


def compute_breakdown(preds, targets, tickers, ticker_to_sector):
    df = pd.DataFrame({'pred': preds, 'target': targets, 'ticker': tickers})
    df['sector'] = df['ticker'].map(ticker_to_sector).fillna('Unknown')

    overall = metrics_for_group(df['target'].values, df['pred'].values)

    per_sector = {}
    for sector, grp in df.groupby('sector'):
        m = metrics_for_group(grp['target'].values, grp['pred'].values)
        if m:
            m['num_tickers'] = int(grp['ticker'].nunique())
            per_sector[sector] = m

    per_ticker = {}
    for ticker, grp in df.groupby('ticker'):
        m = metrics_for_group(grp['target'].values, grp['pred'].values)
        if m:
            per_ticker[ticker] = m

    return {'overall': overall, 'per_sector': per_sector, 'per_ticker': per_ticker}


def main():
    set_seed(SEED)

    # Load sector map once
    raw = pd.read_parquet(PARQUET)[['Ticker', 'Sector']].drop_duplicates()
    ticker_to_sector = dict(zip(raw['Ticker'], raw['Sector']))

    all_results = {}
    best_mcc = -2.0
    best_config = None

    model_configs = [
        ('logistic_regression', 'Logistic Regression'),
        ('lstm',                'LSTM'),
        ('mlp',                 'MLP'),
    ]

    for fs_key in FEATURE_SETS:
        train_ds, val_ds, test_ds, info = build_datasets(
            PARQUET, feature_set=fs_key, window_size=5
        )
        n_feat = info['num_features']
        all_results[fs_key] = {}

        for model_name, model_label in model_configs:
            save_dir = f"results/phase1_baselines/{fs_key}/{model_name}"
            print(f"  {fs_key} | {model_label} ...", end=' ', flush=True)

            if model_name == 'logistic_regression':
                preds, targets, tickers = predict_lr(train_ds, test_ds)

            else:
                pt_path = os.path.join(save_dir, 'best_model.pt')
                if not os.path.exists(pt_path):
                    print("SKIP (no checkpoint)")
                    continue
                if model_name == 'lstm':
                    model = LSTMBaseline(input_dim=n_feat, hidden_dim=64,
                                        num_layers=2, dropout=0.2).to(DEVICE)
                else:
                    model = MLPBaseline(input_dim=n_feat, hidden_dim=128,
                                       dropout=0.2).to(DEVICE)
                model.load_state_dict(torch.load(pt_path, map_location=DEVICE,
                                                 weights_only=True))
                preds, targets, tickers = predict_torch(model, test_ds)

            bd = compute_breakdown(preds, targets, tickers, ticker_to_sector)
            all_results[fs_key][model_name] = {
                'label':     model_label,
                'breakdown': bd,
            }

            mcc = bd['overall']['mcc'] if bd['overall'] else -2.0
            print(f"MCC={mcc:+.4f}")
            if mcc > best_mcc:
                best_mcc, best_config = mcc, f"{model_label} on {fs_key}"

    print(f"\nBest overall: {best_config}  MCC={best_mcc:+.4f}")

    # Save
    out_dir = 'results/phase1_baselines'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'breakdown.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Per-sector summary for best model
    print("\n--- Per-Sector Results (best config) ---")
    best_fs, best_model = None, None
    for fs_key, models in all_results.items():
        for mn, v in models.items():
            if v['breakdown']['overall'] and v['breakdown']['overall']['mcc'] == best_mcc:
                best_fs, best_model = fs_key, mn
    if best_fs:
        bd = all_results[best_fs][best_model]['breakdown']
        for sector, m in sorted(bd['per_sector'].items(), key=lambda x: x[1]['mcc'], reverse=True):
            print(f"  {sector:<22s}  n={m['n_samples']:>4d} ({m['num_tickers']} tickers)  "
                  f"Acc={m['accuracy']:.2%}  MCC={m['mcc']:+.4f}")

    print(f"\nSaved to {out_dir}/breakdown.json")


if __name__ == '__main__':
    main()
