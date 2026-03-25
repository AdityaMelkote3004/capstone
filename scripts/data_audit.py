"""
Phase 0 — Data Audit & Preprocessing Validation

Generates all artifacts needed for Section 4 of the research paper:
  - data_statistics.json    (dataset stats, missing data, splits, coverage)
  - missing_data_heatmap.png
  - target_distribution.png
  - tweet_coverage.png
  - per_sector_target_distribution.png
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.data.stocknet_dataset import (
    load_and_clean, split_per_ticker,
    PRICE_FEATURES, FUNDAMENTAL_FEATURES, TWEET_FEATURES,
    ALL_FUNDAMENTAL_COLS, FEATURE_SETS, SECTOR_MAP,
)

PARQUET  = 'dataset/stocknet_final_modeling_set.parquet'
OUT_DIR  = 'results/phase0_data_audit'


def audit_basic_stats(df):
    """Dataset-level statistics."""
    return {
        'total_rows':       int(len(df)),
        'num_tickers':      int(df['Ticker'].nunique()),
        'num_sectors':      int(df['Sector'].nunique()),
        'total_columns':    int(len(df.columns)),
        'date_range':       {'start': str(df['Date'].min()), 'end': str(df['Date'].max())},
        'rows_per_ticker':  {
            'mean': round(float(df.groupby('Ticker').size().mean()), 1),
            'min':  int(df.groupby('Ticker').size().min()),
            'max':  int(df.groupby('Ticker').size().max()),
        },
        'tickers_per_sector': {
            sector: int(count)
            for sector, count in df.groupby('Sector')['Ticker'].nunique().sort_values(ascending=False).items()
        },
        'sectors': sorted(df['Sector'].unique().tolist()),
        'tickers': sorted(df['Ticker'].unique().tolist()),
    }


def audit_target(df):
    """Target class distribution."""
    counts = df['Target'].value_counts().sort_index()
    total = len(df)
    return {
        'target_distribution': {str(k): int(v) for k, v in counts.items()},
        'target_balance': round(float(counts.get(1, 0) / total), 4),
        'target_pct': {str(k): round(float(v / total * 100), 2) for k, v in counts.items()},
    }


def audit_missing_data(raw_df):
    """Missing data analysis on RAW (uncleaned) data."""
    all_features = PRICE_FEATURES + ALL_FUNDAMENTAL_COLS + TWEET_FEATURES
    missing = {}
    for col in all_features:
        if col in raw_df.columns:
            n_miss = int(raw_df[col].isnull().sum())
            n_inf = int(np.isinf(raw_df[col]).sum()) if raw_df[col].dtype in ['float64', 'float32'] else 0
            missing[col] = {
                'n_missing':   n_miss,
                'n_inf':       n_inf,
                'pct_missing': round(float((n_miss + n_inf) / len(raw_df) * 100), 2),
                'group':       'Price/Technical' if col in PRICE_FEATURES
                               else 'Fundamental' if col in ALL_FUNDAMENTAL_COLS
                               else 'Tweet',
            }
    return missing


def audit_split_sizes(df, train_ratio=0.8):
    """Per-ticker 80/20 chronological split stats."""
    train_df, test_df = split_per_ticker(df, train_ratio)
    total = len(df)
    return {
        'split_sizes': {
            'train':     int(len(train_df)),
            'val':       0,  # No separate val — carved from train during training
            'test':      int(len(test_df)),
            'train_pct': round(float(len(train_df) / total * 100), 1),
            'val_pct':   0.0,
            'test_pct':  round(float(len(test_df) / total * 100), 1),
        },
        'split_method': 'Per-ticker chronological 80/20',
        'train_date_range': {
            'start': str(train_df['Date'].min()),
            'end':   str(train_df['Date'].max()),
        },
        'test_date_range': {
            'start': str(test_df['Date'].min()),
            'end':   str(test_df['Date'].max()),
        },
    }


def audit_tweet_coverage(df):
    """Tweet text coverage analysis."""
    has_company = df['Company_Texts'].notna() & (df['Company_Texts'] != '') & (df['Company_Texts'] != 'nan')
    has_event = df['Event_Texts'].notna() & (df['Event_Texts'] != '') & (df['Event_Texts'] != 'nan')
    has_any = has_company | has_event

    return {
        'tweet_coverage': {
            'company_text_pct': round(float(has_company.mean() * 100), 2),
            'event_text_pct':   round(float(has_event.mean() * 100), 2),
            'any_text_pct':     round(float(has_any.mean() * 100), 2),
            'company_text_rows': int(has_company.sum()),
            'event_text_rows':   int(has_event.sum()),
            'any_text_rows':     int(has_any.sum()),
        },
        'tweet_count_stats': {
            col: {
                'mean':   round(float(df[col].mean()), 2),
                'median': round(float(df[col].median()), 2),
                'max':    int(df[col].max()),
                'zero_pct': round(float((df[col] == 0).mean() * 100), 2),
            }
            for col in TWEET_FEATURES if col in df.columns
        },
    }


def audit_feature_sets():
    """Feature set definitions and dimensions."""
    return {
        'feature_sets': {
            name: {'num_features': len(cols), 'columns': cols}
            for name, cols in FEATURE_SETS.items()
        }
    }


def audit_infinities(raw_df):
    """Check for infinities in raw data."""
    issues = {}
    for col in raw_df.select_dtypes(include=[np.number]).columns:
        n_inf = int(np.isinf(raw_df[col]).sum())
        if n_inf > 0:
            issues[col] = n_inf
    return {'infinity_issues': issues}


def audit_per_sector_target(df):
    """Target distribution per sector."""
    result = {}
    for sector, grp in df.groupby('Sector'):
        counts = grp['Target'].value_counts().sort_index()
        total = len(grp)
        result[sector] = {
            'total': int(total),
            'up': int(counts.get(1, 0)),
            'down': int(counts.get(0, 0)),
            'up_pct': round(float(counts.get(1, 0) / total * 100), 2),
        }
    return {'per_sector_target': result}


# ── Plotting ──────────────────────────────────────────────

def plot_missing_data_heatmap(missing, out_path):
    """Missing data heatmap by feature group."""
    features = []
    pcts = []
    groups = []
    for feat, info in sorted(missing.items(), key=lambda x: x[1]['pct_missing'], reverse=True):
        features.append(feat)
        pcts.append(info['pct_missing'])
        groups.append(info['group'])

    colors = {'Price/Technical': '#3498db', 'Fundamental': '#e74c3c', 'Tweet': '#2ecc71'}
    bar_colors = [colors.get(g, '#95a5a6') for g in groups]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(features)), pcts, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('% Missing / Infinite', fontsize=11)
    ax.set_title('Missing & Infinite Data by Feature (Raw Dataset)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for bar, pct in zip(bars, pcts):
        if pct > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}%', va='center', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_target_distribution(df, stats, out_path):
    """Target distribution — overall and per-sector."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall
    target = stats['target_distribution']
    labels = ['Down (0)', 'Up (1)']
    values = [target['0'], target['1']]
    colors = ['#e74c3c', '#2ecc71']
    axes[0].bar(labels, values, color=colors, edgecolor='white', width=0.5)
    axes[0].set_title('Overall Target Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, (v, pct) in enumerate(zip(values, [stats['target_pct']['0'], stats['target_pct']['1']])):
        axes[0].text(i, v + 100, f'{v:,}\n({pct}%)', ha='center', fontsize=10)

    # Per-sector
    per_sector = df.groupby('Sector')['Target'].mean().sort_values()
    bars = axes[1].barh(per_sector.index, per_sector.values, color='#9b59b6', edgecolor='white')
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Balanced (50%)')
    axes[1].set_xlabel('% Up (Target=1)')
    axes[1].set_title('Target Balance by Sector', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    for bar, val in zip(bars, per_sector.values):
        axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1%}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_tweet_coverage(df, out_path):
    """Tweet coverage analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Coverage per sector
    has_company = df['Company_Texts'].notna() & (df['Company_Texts'] != '') & (df['Company_Texts'] != 'nan')
    has_event = df['Event_Texts'].notna() & (df['Event_Texts'] != '') & (df['Event_Texts'] != 'nan')

    sector_coverage = df.groupby('Sector').apply(
        lambda g: pd.Series({
            'Company': (g['Company_Texts'].notna() & (g['Company_Texts'] != '') & (g['Company_Texts'] != 'nan')).mean() * 100,
            'Event': (g['Event_Texts'].notna() & (g['Event_Texts'] != '') & (g['Event_Texts'] != 'nan')).mean() * 100,
        }), include_groups=False
    ).sort_values('Company', ascending=True)

    x = range(len(sector_coverage))
    axes[0].barh([i - 0.15 for i in x], sector_coverage['Company'], height=0.3, color='#3498db', label='Company Tweets')
    axes[0].barh([i + 0.15 for i in x], sector_coverage['Event'], height=0.3, color='#e67e22', label='Event Tweets')
    axes[0].set_yticks(list(x))
    axes[0].set_yticklabels(sector_coverage.index, fontsize=9)
    axes[0].set_xlabel('% of Trading Days with Text')
    axes[0].set_title('Tweet Text Coverage by Sector', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    # Tweet count distribution
    for col, color, label in [
        ('Company_Tweet_Count', '#3498db', 'Company'),
        ('Event_Tweet_Count', '#e67e22', 'Event'),
    ]:
        nonzero = df[df[col] > 0][col]
        if len(nonzero) > 0:
            axes[1].hist(nonzero.clip(upper=nonzero.quantile(0.95)), bins=30,
                         alpha=0.6, color=color, label=f'{label} (n={len(nonzero):,})')
    axes[1].set_xlabel('Tweet Count (per day, clipped at 95th pct)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Tweet Count Distribution (non-zero days)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_sector_target(per_sector_target, out_path):
    """Per-sector target distribution bar chart."""
    sectors = sorted(per_sector_target.keys())
    up_pcts = [per_sector_target[s]['up_pct'] for s in sectors]
    down_pcts = [100 - p for p in up_pcts]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(sectors))
    ax.bar(x, down_pcts, color='#e74c3c', label='Down', width=0.6)
    ax.bar(x, up_pcts, bottom=down_pcts, color='#2ecc71', label='Up', width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace('_', ' ') for s in sectors], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Percentage')
    ax.set_title('Target Distribution per Sector', fontsize=13, fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    for i, pct in enumerate(up_pcts):
        ax.text(i, 50, f'{pct:.1f}%', ha='center', va='center', fontsize=8,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Phase 0: Data Audit & Preprocessing Validation")
    print("=" * 65)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load raw data (before cleaning) for missing data analysis
    print("\nLoading raw dataset...")
    raw_df = pd.read_parquet(PARQUET)
    raw_df = raw_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    print(f"  Raw: {len(raw_df):,} rows × {len(raw_df.columns)} columns")

    # Load cleaned data
    print("Loading and cleaning dataset...")
    df = load_and_clean(PARQUET)
    print(f"  Clean: {len(df):,} rows × {len(df.columns)} columns")

    # Gather all statistics
    print("\nComputing statistics...")
    stats = {}
    stats.update(audit_basic_stats(df))
    stats.update(audit_target(df))
    stats['missing_data'] = audit_missing_data(raw_df)
    stats.update(audit_split_sizes(df))
    stats.update(audit_tweet_coverage(df))
    stats.update(audit_feature_sets())
    stats.update(audit_infinities(raw_df))
    stats.update(audit_per_sector_target(df))

    # Data quality issues found and resolved
    stats['data_quality_fixes'] = {
        'Volume_Change_infinities': {
            'issue': 'Infinity values in Volume_Change column',
            'count': int(np.isinf(raw_df['Volume_Change']).sum()) if 'Volume_Change' in raw_df.columns else 0,
            'fix': 'Replaced inf/-inf with 0.0',
        },
        'null_values': {
            'issue': 'Null values in technical and tweet features',
            'fix': 'Filled with 0.0',
        },
        'fundamental_nulls': {
            'issue': 'Missing SEC EDGAR fundamentals (~40-50%)',
            'fix': 'Forward-fill within each ticker, then zero-fill remaining',
        },
        'normalization': {
            'method': 'Z-score normalization using training set statistics only',
            'no_leakage': True,
        },
    }

    # Save statistics JSON
    stats_path = os.path.join(OUT_DIR, 'data_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n  Saved: {stats_path}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_missing_data_heatmap(stats['missing_data'], os.path.join(OUT_DIR, 'missing_data_heatmap.png'))
    plot_target_distribution(df, stats, os.path.join(OUT_DIR, 'target_distribution.png'))
    plot_tweet_coverage(df, os.path.join(OUT_DIR, 'tweet_coverage.png'))
    plot_per_sector_target(stats['per_sector_target'], os.path.join(OUT_DIR, 'per_sector_target_distribution.png'))

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  Data Audit Summary")
    print(f"{'=' * 65}")
    print(f"  Dataset: {stats['total_rows']:,} rows × {stats['total_columns']} columns")
    print(f"  Tickers: {stats['num_tickers']} across {stats['num_sectors']} sectors")
    print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"  Target balance: {stats['target_balance']:.1%} up")
    print(f"  Tweet coverage: {stats['tweet_coverage']['any_text_pct']:.1f}% of trading days")
    print(f"  Split: {stats['split_sizes']['train_pct']}% train / {stats['split_sizes']['test_pct']}% test")
    print(f"\n  Feature sets:")
    for name, fs in stats['feature_sets'].items():
        print(f"    {name}: {fs['num_features']} features")

    inf_issues = stats['infinity_issues']
    if inf_issues:
        print(f"\n  Infinity issues found (in raw data):")
        for col, count in inf_issues.items():
            print(f"    {col}: {count} infinite values -> fixed")

    print(f"\n  All artifacts saved to {OUT_DIR}/")
    print(f"  Ready for Section 4: Experimental Setup in the paper.")


if __name__ == '__main__':
    main()
