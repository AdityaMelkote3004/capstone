"""
Phase 1 (v2) — Baseline Modeling on the corrected dataset (dataset/final/).

Same protocol as scripts/train_baselines.py (global date split, 4 feature sets,
3 models, Accuracy/F1/MCC/AUC), pointed at the reprocessed dataset that fixes:
  - authoritative sector mapping (StockTable, not a hand-typed dict)
  - correct SEC EDGAR ticker/CIK list (the actual 87 tickers)
  - TotalLiabilities tag bug (dropped bad LiabilitiesAndStockholdersEquity fallback)
  - fundamental missingness flags + Days_Since_Filing + QoQ growth features
  - tweet-to-trading-day alignment respecting market close (no post-close leakage)

CAVEAT (see src/data/stocknet_dataset.py StockNetDataset.skipped_gap_windows):
Buffer-zone days (near-zero return, dropped from Target labeling) were removed
from the saved dataset entirely, not just from labeling. That means a strict
"5 real consecutive trading days" window check throws away ~87% of samples
(verified empirically: 26,168 candidate windows -> only 1,941 pass for FS1
train). That's too little to train on, so this run uses strict_windows=False,
which matches the *same* windowing convention as the original Phase 1
baseline (allows a window to silently splice over dropped buffer-zone days).
Results below are only a fair, apples-to-apples comparison to the original
Phase 1 numbers under that convention -- NOT a claim that windows here are
exactly 5 consecutive trading days. The real fix requires regenerating the
dataset so buffer-zone rows are kept (Target=NaN) instead of deleted, which
needs a notebook change + Colab rerun.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.stocknet_dataset import build_datasets
from src.models.baselines import LSTMBaseline, MLPBaseline
from src.training.trainer import Trainer, compute_metrics
from src.utils.seed import set_seed
from sklearn.linear_model import LogisticRegression

PARQUET   = 'dataset/final/stocknet_final_modeling_set.parquet'
SEED      = 42
EPOCHS    = 50
PATIENCE  = 10
STRICT_WINDOWS = False  # see module docstring


def run_logistic_regression(train_ds, test_ds, save_dir):
    X_train, y_train = train_ds.to_numpy(use_window=False)
    X_test,  y_test  = test_ds.to_numpy(use_window=False)

    model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, preds, probs)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({'test': m}, f, indent=2)
    return m


def run_lstm(train_ds, val_ds, test_ds, input_dim, save_dir):
    set_seed(SEED)
    model = LSTMBaseline(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    trainer = Trainer(model, train_ds, val_ds, test_ds, lr=0.001, save_dir=save_dir)
    return trainer.train(num_epochs=EPOCHS, patience=PATIENCE)


def run_mlp(train_ds, val_ds, test_ds, input_dim, save_dir):
    set_seed(SEED)
    model = MLPBaseline(input_dim=input_dim, hidden_dim=128, dropout=0.2)
    trainer = Trainer(model, train_ds, val_ds, test_ds, lr=0.001, save_dir=save_dir)
    return trainer.train(num_epochs=EPOCHS, patience=PATIENCE)


def fmt(m):
    return (f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
            f"MCC={m['mcc']:.4f}  AUC={m.get('auc', 0.5):.4f}")


def main():
    set_seed(SEED)
    all_results = {}

    feature_set_labels = {
        'FS1_Price':              'Price Only (14 technical)',
        'FS2_Price_Fundamentals': 'Price + Fundamentals (8 EDGAR)',
        'FS3_Price_Tweets':       'Price + Tweet Counts (3 cols)',
        'FS4_Full_Structured':    'Price + Fundamentals + Tweets (Full)',
    }

    for fs_key, fs_label in feature_set_labels.items():
        print(f"\n{'='*65}")
        print(f"  Feature Set: {fs_label}")
        print(f"{'='*65}")

        train_ds, val_ds, test_ds, info = build_datasets(
            PARQUET, feature_set=fs_key, window_size=5, strict_windows=STRICT_WINDOWS
        )
        n_feat = info['num_features']
        print(f"  Features={n_feat} | Train={info['train_size']} | "
              f"Val={info['val_size']} | Test={info['test_size']} | "
              f"Tickers={info['num_tickers']}")

        fs_results = {}

        print(f"\n  [1/3] Logistic Regression")
        save_dir = f"results/phase1_baselines_v2/{fs_key}/logistic_regression"
        m = run_logistic_regression(train_ds, test_ds, save_dir)
        fs_results['Logistic Regression'] = m
        print(f"    {fmt(m)}")

        print(f"\n  [2/3] LSTM (sliding window, W=5)")
        save_dir = f"results/phase1_baselines_v2/{fs_key}/lstm"
        m = run_lstm(train_ds, val_ds, test_ds, n_feat, save_dir)
        fs_results['LSTM'] = m
        print(f"    {fmt(m)}")

        print(f"\n  [3/3] MLP")
        save_dir = f"results/phase1_baselines_v2/{fs_key}/mlp"
        m = run_mlp(train_ds, val_ds, test_ds, n_feat, save_dir)
        fs_results['MLP'] = m
        print(f"    {fmt(m)}")

        all_results[fs_key] = {
            'label': fs_label,
            'num_features': n_feat,
            'train_size': info['train_size'],
            'val_size': info['val_size'],
            'test_size': info['test_size'],
            'models': fs_results,
        }

    print(f"\n\n{'='*75}")
    print(f"  AGGREGATED RESULTS — 3 Models x 4 Feature Sets (Test Set, corrected dataset)")
    print(f"{'='*75}")

    models = ['Logistic Regression', 'LSTM', 'MLP']
    header = f"  {'Feature Set':<35s}"
    for model in models:
        header += f"  {model:<22s}"
    print(header)
    print(f"  {'':35s}" + "  " + ("  Acc    F1   MCC   AUC" * len(models)))
    print(f"  {'-'*73}")

    for fs_key, res in all_results.items():
        row = f"  {res['label']:<35s}"
        for model in models:
            m = res['models'].get(model, {})
            row += (f"  {m.get('accuracy',0):.3f} {m.get('f1',0):.3f} "
                    f"{m.get('mcc',0):+.3f} {m.get('auc',0.5):.3f}")
        print(row)

    os.makedirs('results/phase1_baselines_v2', exist_ok=True)
    summary_flat = {}
    for fs_key, res in all_results.items():
        for model, m in res['models'].items():
            key = f"{model} | {res['label']}"
            summary_flat[key] = {
                'accuracy':  m.get('accuracy', 0),
                'f1':        m.get('f1', 0),
                'mcc':       m.get('mcc', 0),
                'auc':       m.get('auc', 0.5),
                'n_samples': m.get('n_samples', 0),
                'confusion_matrix': m.get('confusion_matrix', []),
                'feature_set': res['label'],
                'model': model,
            }
    with open('results/phase1_baselines_v2/summary.json', 'w') as f:
        json.dump(summary_flat, f, indent=2)
    with open('results/phase1_baselines_v2/full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Saved to results/phase1_baselines_v2/")


if __name__ == '__main__':
    main()
