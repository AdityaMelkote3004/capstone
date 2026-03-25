"""
Phase 1 — Per-Ticker & Per-Sector Evaluation of All Baselines

Loads saved model weights from results/phase1_baselines/{feature_set}/{model}/
and produces detailed breakdowns for the research paper:
  - Per-sector accuracy/F1/MCC for each model x feature set
  - Per-ticker accuracy/MCC for each model x feature set
  - Ablation analysis: which feature set adds the most?
  - Saves breakdown.json and visualization PNGs
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                              matthews_corrcoef, roc_auc_score, confusion_matrix)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.stocknet_dataset import build_datasets, FEATURE_SETS
from src.models.baselines import LSTMBaseline, MLPBaseline
from src.training.trainer import compute_metrics
from src.utils.seed import set_seed

PARQUET = 'dataset/stocknet_final_modeling_set.parquet'
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED    = 42

MODELS = ['logistic_regression', 'lstm', 'mlp']
MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'lstm': 'LSTM',
    'mlp': 'MLP',
}


def per_group_metrics(y_true, y_pred, min_samples=5):
    """Compute metrics, return None if too few samples."""
    if len(y_true) < min_samples:
        return None
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'f1':       round(float(f1_score(y_true, y_pred, average='binary', zero_division=0)), 4),
        'mcc':      round(float(matthews_corrcoef(y_true, y_pred)), 4),
        'n_samples': int(len(y_true)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def predict_sklearn(train_ds, test_ds):
    """Train LR and predict (sklearn has no .pt to load)."""
    X_train, y_train = train_ds.to_numpy(use_window=False)
    X_test, y_test = test_ds.to_numpy(use_window=False)
    model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test


def predict_neural(model, test_ds, model_type='lstm'):
    """Run inference with a saved PyTorch model."""
    model.eval()
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == 'lstm':
                logits = model(window=batch['window'].to(DEVICE))
            else:  # mlp
                logits = model(flat=batch['flat'].to(DEVICE))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(batch['target'].numpy())
    return np.array(all_preds), np.array(all_targets)


def compute_breakdown(preds, targets, test_ds, raw_df):
    """Compute per-ticker and per-sector breakdowns."""
    tickers = [test_ds.samples[i]['ticker'] for i in range(len(test_ds))]
    df = pd.DataFrame({'pred': preds, 'target': targets, 'ticker': tickers})

    ticker_sector = raw_df[['Ticker', 'Sector']].drop_duplicates()
    ticker_sector = dict(zip(ticker_sector['Ticker'], ticker_sector['Sector']))
    df['sector'] = df['ticker'].map(ticker_sector)

    # Overall
    overall = per_group_metrics(df['target'].values, df['pred'].values)
    overall['scope'] = f'ALL {df["ticker"].nunique()} TICKERS'

    # Per-sector
    per_sector = {}
    for sector, grp in df.groupby('sector'):
        m = per_group_metrics(grp['target'].values, grp['pred'].values)
        if m:
            m['num_tickers'] = int(grp['ticker'].nunique())
            per_sector[sector] = m

    # Per-ticker
    per_ticker = {}
    for ticker, grp in df.groupby('ticker'):
        m = per_group_metrics(grp['target'].values, grp['pred'].values)
        if m:
            per_ticker[ticker] = m

    return {'overall': overall, 'per_sector': per_sector, 'per_ticker': per_ticker}


def plot_sector_comparison(all_breakdowns, out_dir):
    """Bar chart comparing models across sectors for best feature set."""
    # Find best feature set by average MCC across models
    best_fs = None
    best_avg_mcc = -1
    for fs_key in FEATURE_SETS:
        mccs = []
        for model in MODELS:
            key = f"{fs_key}/{MODEL_LABELS[model]}"
            if key in all_breakdowns:
                mccs.append(all_breakdowns[key]['overall']['mcc'])
        if mccs and np.mean(mccs) > best_avg_mcc:
            best_avg_mcc = np.mean(mccs)
            best_fs = fs_key

    if not best_fs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sectors = sorted(list(next(iter(all_breakdowns.values()))['per_sector'].keys()))
    x = np.arange(len(sectors))
    width = 0.25

    for metric_idx, (metric, ax) in enumerate(zip(['accuracy', 'mcc'], axes)):
        for i, model in enumerate(MODELS):
            key = f"{best_fs}/{MODEL_LABELS[model]}"
            if key not in all_breakdowns:
                continue
            vals = [all_breakdowns[key]['per_sector'].get(s, {}).get(metric, 0) for s in sectors]
            ax.bar(x + i * width, vals, width, label=MODEL_LABELS[model], alpha=0.85)

        if metric == 'accuracy':
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        elif metric == 'mcc':
            ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='Random')

        ax.set_xticks(x + width)
        ax.set_xticklabels([s.replace('_', ' ') for s in sectors], rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Per-Sector {metric.upper()} ({best_fs})', fontweight='bold')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'per_sector_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_ablation(full_results, out_dir):
    """Ablation bar chart: MCC improvement per feature set relative to FS1."""
    fig, ax = plt.subplots(figsize=(10, 5))

    fs_keys = list(FEATURE_SETS.keys())
    x = np.arange(len(MODELS))
    width = 0.2

    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    for i, fs_key in enumerate(fs_keys):
        mccs = []
        for model in MODELS:
            path = f"results/phase1_baselines/{fs_key}/{model}/metrics.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                mccs.append(data['test']['mcc'])
            else:
                mccs.append(0)
        ax.bar(x + i * width, mccs, width, label=fs_key.replace('_', ' '), color=colors[i], alpha=0.85)

    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel('MCC')
    ax.set_title('Ablation: MCC by Model x Feature Set', fontweight='bold')
    ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    path = os.path.join(out_dir, 'ablation_mcc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(full_results, out_dir):
    """Confusion matrices for all 12 experiments in a 3x4 grid."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fs_keys = list(FEATURE_SETS.keys())
    fs_short = ['FS1: Price', 'FS2: +Fund', 'FS3: +Tweet', 'FS4: Full']

    for row, model in enumerate(MODELS):
        for col, (fs_key, fs_label) in enumerate(zip(fs_keys, fs_short)):
            ax = axes[row][col]
            path = f"results/phase1_baselines/{fs_key}/{model}/metrics.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                cm = np.array(data['test']['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                            cbar=False)
                acc = data['test']['accuracy']
                mcc = data['test']['mcc']
                ax.set_title(f'{MODEL_LABELS[model]}\n{fs_label}\nAcc={acc:.3f} MCC={mcc:+.3f}',
                             fontsize=9, fontweight='bold')
            else:
                ax.set_title(f'{MODEL_LABELS[model]}\n{fs_label}\n(missing)', fontsize=9)
                ax.axis('off')

            if col == 0:
                ax.set_ylabel('Actual')
            if row == 2:
                ax.set_xlabel('Predicted')

    plt.suptitle('Confusion Matrices: 3 Models x 4 Feature Sets (Test Set, 87 Tickers)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrices_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    set_seed(SEED)
    raw_df = pd.read_parquet(PARQUET)
    out_dir = 'results/phase1_baselines/per_ticker_breakdown'
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("  Phase 1 Evaluation: Per-Ticker & Per-Sector Breakdown")
    print("=" * 65)

    all_breakdowns = {}

    for fs_key in FEATURE_SETS:
        print(f"\n--- Feature Set: {fs_key} ---")
        train_ds, test_ds, info = build_datasets(
            PARQUET, feature_set=fs_key, window_size=5, train_ratio=0.8
        )
        n_feat = info['num_features']

        for model_name in MODELS:
            label = MODEL_LABELS[model_name]
            save_dir = f"results/phase1_baselines/{fs_key}/{model_name}"

            if model_name == 'logistic_regression':
                print(f"  {label}: re-fitting (no .pt for sklearn)...")
                preds, targets = predict_sklearn(train_ds, test_ds)

            elif model_name == 'lstm':
                pt_path = os.path.join(save_dir, 'best_model.pt')
                if not os.path.exists(pt_path):
                    print(f"  {label}: weights not found, skipping")
                    continue
                print(f"  {label}: loading {pt_path}")
                model = LSTMBaseline(input_dim=n_feat, hidden_dim=64,
                                     num_layers=2, dropout=0.2).to(DEVICE)
                model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
                preds, targets = predict_neural(model, test_ds, 'lstm')

            elif model_name == 'mlp':
                pt_path = os.path.join(save_dir, 'best_model.pt')
                if not os.path.exists(pt_path):
                    print(f"  {label}: weights not found, skipping")
                    continue
                print(f"  {label}: loading {pt_path}")
                model = MLPBaseline(input_dim=n_feat, hidden_dim=128,
                                    dropout=0.2).to(DEVICE)
                model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
                preds, targets = predict_neural(model, test_ds, 'mlp')

            bd = compute_breakdown(preds, targets, test_ds, raw_df)
            key = f"{fs_key}/{label}"
            all_breakdowns[key] = bd

            o = bd['overall']
            print(f"    Overall: Acc={o['accuracy']:.4f} F1={o['f1']:.4f} "
                  f"MCC={o['mcc']:.4f} (n={o['n_samples']})")

    # Save full breakdown
    bd_path = os.path.join(out_dir, 'breakdown.json')
    with open(bd_path, 'w') as f:
        json.dump(all_breakdowns, f, indent=2)
    print(f"\n  Saved breakdown: {bd_path}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_sector_comparison(all_breakdowns, out_dir)
    plot_ablation(None, out_dir)
    plot_confusion_matrices(None, out_dir)

    # Print ablation summary
    print(f"\n{'=' * 65}")
    print(f"  ABLATION ANALYSIS: Which modality adds the most?")
    print(f"{'=' * 65}")
    print(f"  {'Model':<22s} {'FS1 (Price)':>12} {'FS2 (+Fund)':>12} "
          f"{'FS3 (+Tweet)':>12} {'FS4 (Full)':>12}")
    print(f"  {'-' * 72}")

    for model in MODELS:
        row = f"  {MODEL_LABELS[model]:<22s}"
        for fs_key in FEATURE_SETS:
            path = f"results/phase1_baselines/{fs_key}/{model}/metrics.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                mcc = data['test']['mcc']
                row += f" {mcc:>+12.4f}"
            else:
                row += f" {'N/A':>12s}"
        print(row)

    # Per-sector summary for best model
    best_key = max(all_breakdowns.keys(),
                   key=lambda k: all_breakdowns[k]['overall']['mcc'])
    print(f"\n  Best configuration: {best_key}")
    print(f"  Per-sector breakdown:")
    bd = all_breakdowns[best_key]
    print(f"  {'Sector':<22s} {'Tickers':>7} {'Samples':>8} {'Acc':>8} {'MCC':>8}")
    print(f"  {'-' * 55}")
    for sector, m in sorted(bd['per_sector'].items()):
        print(f"  {sector:<22s} {m['num_tickers']:>7d} {m['n_samples']:>8d} "
              f"{m['accuracy']:>8.2%} {m['mcc']:>8.4f}")


if __name__ == '__main__':
    main()
