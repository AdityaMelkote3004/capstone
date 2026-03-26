"""
Phase 1 — Baseline Modeling Strategy (per flowchart)
  - Global date split: Train 2014-01-02->2015-03-31, Val->2015-07-31, Test->2015-12-31
  - 4 Feature Sets (modality ablation)
  - 3 Models: LSTM, Logistic Regression, MLP
  - Metrics: Accuracy, F1, MCC, AUC
  - Results: 3 models x 4 feature sets = 12 cells
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                              matthews_corrcoef, roc_auc_score, confusion_matrix)

from src.data.stocknet_dataset import build_datasets, FEATURE_SETS
from src.models.baselines import LSTMBaseline, MLPBaseline
from src.training.trainer import Trainer, compute_metrics
from src.utils.seed import set_seed

PARQUET   = 'dataset/stocknet_final_modeling_set.parquet'
SEED      = 42
EPOCHS    = 50
PATIENCE  = 10


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
    model = LSTMBaseline(input_dim=input_dim, hidden_dim=64,
                         num_layers=2, dropout=0.2)
    trainer = Trainer(model, train_ds, val_ds, test_ds,
                      lr=0.001, save_dir=save_dir)
    return trainer.train(num_epochs=EPOCHS, patience=PATIENCE)


def run_mlp(train_ds, val_ds, test_ds, input_dim, save_dir):
    set_seed(SEED)
    model = MLPBaseline(input_dim=input_dim, hidden_dim=128, dropout=0.2)
    trainer = Trainer(model, train_ds, val_ds, test_ds,
                      lr=0.001, save_dir=save_dir)
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
            PARQUET, feature_set=fs_key, window_size=5
        )
        n_feat = info['num_features']
        print(f"  Features={n_feat} | Train={info['train_size']} | "
              f"Val={info['val_size']} | Test={info['test_size']} | "
              f"Tickers={info['num_tickers']}")

        fs_results = {}

        # ── Logistic Regression ──────────────────────────────
        print(f"\n  [1/3] Logistic Regression")
        save_dir = f"results/phase1_baselines/{fs_key}/logistic_regression"
        m = run_logistic_regression(train_ds, test_ds, save_dir)
        fs_results['Logistic Regression'] = m
        print(f"    {fmt(m)}")

        # ── LSTM ────────────────────────────────────────────
        print(f"\n  [2/3] LSTM (sliding window, W=5)")
        save_dir = f"results/phase1_baselines/{fs_key}/lstm"
        m = run_lstm(train_ds, val_ds, test_ds, n_feat, save_dir)
        fs_results['LSTM'] = m
        print(f"    {fmt(m)}")

        # ── MLP ─────────────────────────────────────────────
        print(f"\n  [3/3] MLP")
        save_dir = f"results/phase1_baselines/{fs_key}/mlp"
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

    # ── Aggregated results table ───────────────────────────
    print(f"\n\n{'='*75}")
    print(f"  AGGREGATED RESULTS — 3 Models × 4 Feature Sets (Test Set, 87 Tickers)")
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

    # ── Save summary ────────────────────────────────────────
    os.makedirs('results/phase1_baselines', exist_ok=True)

    # Flat summary for dashboard
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
    with open('results/phase1_baselines/summary.json', 'w') as f:
        json.dump(summary_flat, f, indent=2)

    # Full results
    with open('results/phase1_baselines/full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Saved to results/phase1_baselines/")
    print(f"  Total experiments: {len(models) * len(all_results)} "
          f"({len(models)} models × {len(all_results)} feature sets)")


if __name__ == '__main__':
    main()
