"""MMGTFFF -- Results Dashboard (Phase 0 & Phase 1)"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="MMGTFFF Dashboard",
    page_icon="📈",
    layout="wide",
)

# -- Load data ---------------------------------------------------------------

@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)

def safe_load(path, default=None):
    try:
        return load_json(path)
    except Exception:
        return default

stats      = safe_load("results/phase0_data_audit/data_statistics.json", {})
summary    = safe_load("results/phase1_baselines/summary.json", {})
full_results = safe_load("results/phase1_baselines/full_results.json", {})
breakdown  = safe_load("results/phase1_baselines/per_ticker_breakdown/breakdown.json", {})

# -- Sidebar ------------------------------------------------------------------

st.sidebar.title("MMGTFFF")
st.sidebar.caption("Multi-Modal Graph Transformer\nfor Federated Financial Forecasting")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Phase 0: Data Audit",
    "Phase 1: Baselines",
    "Ablation Analysis",
    "Per-Ticker & Sector Analysis",
    "What This Means",
])

# Feature set / model constants
FS_KEYS = ['FS1_Price', 'FS2_Price_Fundamentals', 'FS3_Price_Tweets', 'FS4_Full_Structured']
FS_LABELS = {
    'FS1_Price': 'FS1: Price Only (14)',
    'FS2_Price_Fundamentals': 'FS2: Price + Fund (22)',
    'FS3_Price_Tweets': 'FS3: Price + Tweet (17)',
    'FS4_Full_Structured': 'FS4: Full (25)',
}
FS_SHORT = {
    'FS1_Price': 'Price Only',
    'FS2_Price_Fundamentals': '+ Fundamentals',
    'FS3_Price_Tweets': '+ Tweets',
    'FS4_Full_Structured': 'Full',
}
MODEL_NAMES = ['Logistic Regression', 'LSTM', 'MLP']
MODEL_COLORS = {'Logistic Regression': '#4472C4', 'LSTM': '#ED7D31', 'MLP': '#70AD47'}


# -- Overview -----------------------------------------------------------------

if page == "Overview":
    st.title("MMGTFFF -- Project Dashboard")
    st.markdown("**Multi-Modal Graph Transformer for Federated Financial Forecasting**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers", stats.get("num_tickers", 87))
    col2.metric("Sectors", stats.get("num_sectors", 9))
    col3.metric("Total Samples", f"{stats.get('total_rows', 26603):,}")
    col4.metric("Date Range", "2014 - 2015")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Sources")
        st.markdown("""
| Source | Features | Coverage |
|--------|----------|----------|
| **Yahoo Finance** | Technical (Return, RSI, MACD...) | 100% |
| **StockNet Tweets** | Company & Event text | 43% / 11% |
| **SEC EDGAR** | Fundamentals (EPS, ROA...) | ~60% |
        """)

    with col2:
        st.subheader("Phase Progress")
        phases = {
            "Phase 0 -- Data Audit": "Done",
            "Phase 1 -- Baselines": "Done",
            "Phase 2 -- Modality Encoders": "In Progress",
            "Phase 3 -- Multi-Modal Fusion": "Pending",
            "Phase 4 -- Graph (GAT)": "Pending",
            "Phase 5 -- Federated Learning": "Pending",
            "Phase 6 -- Final Evaluation": "Pending",
        }
        for phase, status in phases.items():
            icon = {"Done": "✅", "In Progress": "⏳", "Pending": "⬜"}.get(status, "⬜")
            st.markdown(f"{icon} &nbsp; {phase}")

    st.divider()

    # 3x4 summary grid
    if full_results:
        st.subheader("Baseline Summary: 3 Models x 4 Feature Sets (Test Set)")
        rows = []
        for fs_key in FS_KEYS:
            fs_data = full_results.get(fs_key, {})
            models_data = fs_data.get('models', {})
            row = {'Feature Set': FS_LABELS.get(fs_key, fs_key)}
            for model in MODEL_NAMES:
                m = models_data.get(model, {})
                mcc = m.get('mcc', 0)
                acc = m.get('accuracy', 0)
                row[f'{model} Acc'] = acc
                row[f'{model} MCC'] = mcc
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({col: "{:.4f}" for col in df.columns if col != 'Feature Set'})
                    .background_gradient(
                        subset=[c for c in df.columns if 'MCC' in c],
                        cmap="RdYlGn", vmin=-0.05, vmax=0.05
                    ),
            use_container_width=True, hide_index=True,
        )

    st.info("**All baselines tested on all 87 tickers combined** "
            "(~4,921 test samples). "
            "No results are single-ticker -- models train and evaluate across the full dataset.")


# -- Phase 0 ------------------------------------------------------------------

elif page == "Phase 0: Data Audit":
    st.title("Phase 0: Data Audit")

    if not stats:
        st.error("data_statistics.json not found. Run `python scripts/data_audit.py` first.")
        st.stop()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{stats['total_rows']:,}")
    col2.metric("Tickers", stats["num_tickers"])
    col3.metric("Sectors", stats["num_sectors"])
    col4.metric("Features", stats["total_columns"])
    col5.metric("Target Balance", f"{stats['target_balance']:.1%}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        target = stats["target_distribution"]
        fig = go.Figure(go.Bar(
            x=["Down (0)", "Up (1)"], y=[target["0"], target["1"]],
            marker_color=["#e74c3c", "#2ecc71"],
            text=[f"{target['0']:,}", f"{target['1']:,}"], textposition="outside",
        ))
        fig.update_layout(yaxis_title="Count", height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tickers per Sector")
        sectors = stats["tickers_per_sector"]
        fig = go.Figure(go.Bar(
            y=list(sectors.keys()), x=list(sectors.values()), orientation="h",
            marker_color="#9b59b6",
            text=list(sectors.values()), textposition="outside",
        ))
        fig.update_layout(xaxis_title="Tickers", height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Missing Data -- Fundamental Features")
    missing = stats.get("missing_data", {})
    fund_rows = [{"Feature": k, "% Missing": v["pct_missing"], "Group": v["group"]}
                 for k, v in missing.items() if v["pct_missing"] > 0]
    if fund_rows:
        fund_df = pd.DataFrame(fund_rows).sort_values("% Missing", ascending=False)
        group_colors = {'Price/Technical': '#3498db', 'Fundamental': '#e74c3c', 'Tweet': '#2ecc71'}
        fig = go.Figure(go.Bar(
            x=fund_df["Feature"], y=fund_df["% Missing"],
            marker_color=[group_colors.get(g, '#95a5a6') for g in fund_df["Group"]],
            text=fund_df["% Missing"].apply(lambda x: f"{x:.1f}%"), textposition="outside",
        ))
        fig.update_layout(yaxis_title="% Missing", xaxis_tickangle=-45,
                          height=350, margin=dict(t=20, b=60))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Technical features have 0% missing. SEC EDGAR fundamentals have ~40-50% gaps (forward-filled within each ticker).")
    else:
        st.success("No missing data after cleaning.")

    st.divider()

    # Tweet coverage
    tweet_cov = stats.get("tweet_coverage", {})
    if tweet_cov:
        st.subheader("Tweet Text Coverage")
        col1, col2, col3 = st.columns(3)
        col1.metric("Company Text", f"{tweet_cov['company_text_pct']:.1f}%")
        col2.metric("Event Text", f"{tweet_cov['event_text_pct']:.1f}%")
        col3.metric("Any Text", f"{tweet_cov['any_text_pct']:.1f}%")

    st.divider()
    st.subheader("Feature Set Definitions")
    fs_info = stats.get("feature_sets", {})
    for name, info in fs_info.items():
        with st.expander(f"{name} ({info['num_features']} features)"):
            st.write(", ".join(info['columns']))

    st.divider()
    st.subheader("Data Quality Fixes Applied")
    fixes = stats.get("data_quality_fixes", {})
    for fix_name, fix_info in fixes.items():
        if isinstance(fix_info, dict) and 'issue' in fix_info:
            st.markdown(f"- **{fix_info['issue']}**: {fix_info['fix']}")
        elif isinstance(fix_info, dict) and 'method' in fix_info:
            st.markdown(f"- **Normalization**: {fix_info['method']}")

    # Chronological split
    st.divider()
    st.subheader("Chronological Split")
    split = stats.get("split_sizes", {})
    if split:
        train_range = stats.get("train_date_range", {})
        test_range = stats.get("test_date_range", {})
        fig = go.Figure(go.Bar(
            x=[f"Train ({train_range.get('start','')[:10]} to {train_range.get('end','')[:10]})",
               f"Test ({test_range.get('start','')[:10]} to {test_range.get('end','')[:10]})"],
            y=[split.get("train", 0), split.get("test", 0)],
            marker_color=["#3498db", "#e74c3c"],
            text=[f"{split.get('train',0):,} ({split.get('train_pct',0)}%)",
                  f"{split.get('test',0):,} ({split.get('test_pct',0)}%)"],
            textposition="outside",
        ))
        fig.update_layout(yaxis_title="Rows", height=340, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Per-ticker chronological split: first 80% of each ticker's dates for train, last 20% for test.")


# -- Phase 1: Baselines -------------------------------------------------------

elif page == "Phase 1: Baselines":
    st.title("Phase 1: Baseline Models")
    st.caption("3 Models x 4 Feature Sets = 12 experiments. Test set across all 87 tickers.")

    if not full_results:
        st.error("full_results.json not found. Run `python scripts/train_baselines.py` first.")
        st.stop()

    # Main results table
    st.subheader("Test Set Results -- 3 x 4 Grid")
    rows = [{"Model": "Random Guess", "Feature Set": "-",
             "Accuracy": 0.5000, "F1": 0.5000, "MCC": 0.0000, "AUC": 0.5000}]

    for fs_key in FS_KEYS:
        fs_data = full_results.get(fs_key, {})
        models_data = fs_data.get('models', {})
        for model in MODEL_NAMES:
            m = models_data.get(model, {})
            rows.append({
                "Model": model,
                "Feature Set": FS_SHORT.get(fs_key, fs_key),
                "Accuracy": m.get('accuracy', 0),
                "F1": m.get('f1', 0),
                "MCC": m.get('mcc', 0),
                "AUC": m.get('auc', 0.5),
            })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "MCC": "{:+.4f}", "AUC": "{:.4f}"})
                .background_gradient(subset=["MCC"], cmap="RdYlGn", vmin=-0.05, vmax=0.05),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # Bar charts: Accuracy and MCC by model for each feature set
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy by Model x Feature Set")
        fig = go.Figure()
        for model in MODEL_NAMES:
            accs = []
            for fs_key in FS_KEYS:
                m = full_results.get(fs_key, {}).get('models', {}).get(model, {})
                accs.append(m.get('accuracy', 0))
            fig.add_trace(go.Bar(
                name=model, x=[FS_SHORT[k] for k in FS_KEYS], y=accs,
                marker_color=MODEL_COLORS[model],
                text=[f"{a:.3f}" for a in accs], textposition="outside",
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="Random (50%)")
        fig.update_layout(barmode='group', yaxis_range=[0.46, 0.54],
                          yaxis_title="Accuracy", height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("MCC by Model x Feature Set")
        fig = go.Figure()
        for model in MODEL_NAMES:
            mccs = []
            for fs_key in FS_KEYS:
                m = full_results.get(fs_key, {}).get('models', {}).get(model, {})
                mccs.append(m.get('mcc', 0))
            fig.add_trace(go.Bar(
                name=model, x=[FS_SHORT[k] for k in FS_KEYS], y=mccs,
                marker_color=MODEL_COLORS[model],
                text=[f"{m:+.4f}" for m in mccs], textposition="outside",
            ))
        fig.add_hline(y=0.0, line_dash="dash", line_color="red",
                      annotation_text="Random (MCC=0)")
        fig.update_layout(barmode='group', yaxis_range=[-0.05, 0.06],
                          yaxis_title="MCC", height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.divider()
    st.subheader("Confusion Matrices")
    fs_choice = st.selectbox("Feature Set", FS_KEYS, format_func=lambda x: FS_LABELS[x])
    cols = st.columns(3)
    for i, model in enumerate(MODEL_NAMES):
        m = full_results.get(fs_choice, {}).get('models', {}).get(model, {})
        cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
        with cols[i]:
            st.markdown(f"**{model}**")
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred Down", "Pred Up"], y=["Actual Down", "Actual Up"],
                text=[[f"{cm[r][c]}" for c in range(2)] for r in range(2)],
                texttemplate="%{text}", colorscale="Blues", showscale=False,
            ))
            fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Acc={m.get('accuracy',0):.4f} | MCC={m.get('mcc',0):+.4f}")


# -- Ablation Analysis ---------------------------------------------------------

elif page == "Ablation Analysis":
    st.title("Ablation Analysis: Which Modality Adds the Most?")
    st.caption("Comparing feature sets to isolate the contribution of fundamentals, tweets, and combined features.")

    if not full_results:
        st.error("full_results.json not found.")
        st.stop()

    # Delta MCC relative to FS1
    st.subheader("MCC Change Relative to FS1 (Price Only)")
    rows = []
    for model in MODEL_NAMES:
        fs1_mcc = full_results.get('FS1_Price', {}).get('models', {}).get(model, {}).get('mcc', 0)
        for fs_key in FS_KEYS[1:]:  # Skip FS1
            m = full_results.get(fs_key, {}).get('models', {}).get(model, {})
            delta = m.get('mcc', 0) - fs1_mcc
            rows.append({
                'Model': model,
                'Feature Set': FS_SHORT[fs_key],
                'FS1 MCC': fs1_mcc,
                'New MCC': m.get('mcc', 0),
                'Delta MCC': delta,
            })

    delta_df = pd.DataFrame(rows)
    st.dataframe(
        delta_df.style.format({
            "FS1 MCC": "{:+.4f}", "New MCC": "{:+.4f}", "Delta MCC": "{:+.4f}"
        }).background_gradient(subset=["Delta MCC"], cmap="RdYlGn", vmin=-0.05, vmax=0.05),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # Grouped bar chart
    fig = go.Figure()
    for model in MODEL_NAMES:
        deltas = []
        for fs_key in FS_KEYS[1:]:
            fs1_mcc = full_results.get('FS1_Price', {}).get('models', {}).get(model, {}).get('mcc', 0)
            new_mcc = full_results.get(fs_key, {}).get('models', {}).get(model, {}).get('mcc', 0)
            deltas.append(new_mcc - fs1_mcc)
        fig.add_trace(go.Bar(
            name=model, x=[FS_SHORT[k] for k in FS_KEYS[1:]], y=deltas,
            marker_color=MODEL_COLORS[model],
            text=[f"{d:+.4f}" for d in deltas], textposition="outside",
        ))
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        barmode='group', yaxis_title="Delta MCC vs FS1",
        title="Impact of Adding Each Modality (MCC change from Price-Only baseline)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Key findings
    st.subheader("Key Findings")

    # Find best overall config
    best_model, best_fs, best_mcc = None, None, -1
    for fs_key in FS_KEYS:
        for model in MODEL_NAMES:
            mcc = full_results.get(fs_key, {}).get('models', {}).get(model, {}).get('mcc', 0)
            if mcc > best_mcc:
                best_mcc = mcc
                best_model = model
                best_fs = fs_key

    st.markdown(f"""
- **Best configuration**: {best_model} on {FS_LABELS.get(best_fs, best_fs)} (MCC = {best_mcc:+.4f})
- **Logistic Regression** consistently outperforms neural models (LSTM, MLP) on this dataset
- **Adding fundamentals** (FS2) and **tweets** (FS3) provides marginal improvement for LR
- **Neural models struggle** to generalize -- they overfit on training data but underperform on test
- All models hover near **random performance** (MCC ~ 0), confirming that structured features alone
  cannot reliably predict next-day stock movement
    """)

    st.info("This result directly motivates Phase 2 (deep encoders) and Phase 4 (graph structure) "
            "-- richer representations and inter-stock relationships are needed to improve beyond random.")


# -- Per-Ticker & Sector -------------------------------------------------------

elif page == "Per-Ticker & Sector Analysis":
    st.title("Per-Ticker & Per-Sector Analysis")
    st.caption("Detailed breakdown across all 87 tickers and 9 sectors.")

    if not breakdown:
        st.error("breakdown.json not found. Run `python scripts/evaluate_baselines.py` first.")
        st.stop()

    model_choices = list(breakdown.keys())
    model_choice = st.selectbox("Select Model x Feature Set", model_choices)
    res = breakdown[model_choice]

    o = res["overall"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers", o["scope"].split()[1] if 'scope' in o else '87')
    col2.metric("Test Samples", f"{o['n_samples']:,}")
    col3.metric("Accuracy", f"{o['accuracy']:.2%}")
    col4.metric("MCC", f"{o['mcc']:+.4f}")

    st.divider()

    # Per-sector chart
    st.subheader("Per-Sector Performance")
    sec = res.get("per_sector", {})
    if sec:
        sec_df = pd.DataFrame([
            {"Sector": k.replace("_", " "), "Accuracy": v["accuracy"],
             "MCC": v["mcc"], "Tickers": v["num_tickers"], "Samples": v["n_samples"]}
            for k, v in sec.items()
        ]).sort_values("MCC", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=sec_df["Sector"], y=sec_df["Accuracy"],
                marker_color=["#2ecc71" if v >= 0.5 else "#e74c3c" for v in sec_df["Accuracy"]],
                text=sec_df["Accuracy"].apply(lambda x: f"{x:.1%}"), textposition="outside",
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(yaxis_title="Accuracy", yaxis_range=[0.44, 0.60],
                              xaxis_tickangle=-30, height=380, margin=dict(t=20, b=80))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Bar(
                x=sec_df["Sector"], y=sec_df["MCC"],
                marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in sec_df["MCC"]],
                text=sec_df["MCC"].apply(lambda x: f"{x:+.3f}"), textposition="outside",
            ))
            fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
            fig.update_layout(yaxis_title="MCC", xaxis_tickangle=-30,
                              height=380, margin=dict(t=20, b=80))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sec_df.style.format({"Accuracy": "{:.2%}", "MCC": "{:+.4f}"}),
                     use_container_width=True, hide_index=True)

    st.divider()

    # Per-ticker scatter
    st.subheader("Per-Ticker Performance (all 87 stocks)")
    tick = res.get("per_ticker", {})
    if tick:
        tick_df = pd.DataFrame([
            {"Ticker": k, "Accuracy": v["accuracy"], "MCC": v["mcc"],
             "Samples": v["n_samples"]}
            for k, v in tick.items()
        ]).sort_values("MCC", ascending=False)

        fig = go.Figure(go.Scatter(
            x=tick_df["Ticker"], y=tick_df["MCC"],
            mode="markers",
            marker=dict(
                color=tick_df["MCC"], colorscale="RdYlGn", size=10,
                colorbar=dict(title="MCC"), cmin=-0.35, cmax=0.35,
            ),
            text=tick_df.apply(
                lambda r: f"{r['Ticker']}<br>Acc: {r['Accuracy']:.1%}<br>MCC: {r['MCC']:+.3f}<br>n={r['Samples']}",
                axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Random")
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="MCC",
                          height=440, margin=dict(t=20, b=80))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 Tickers (by MCC)**")
            st.dataframe(tick_df.head(10)[["Ticker", "Accuracy", "MCC", "Samples"]]
                         .style.format({"Accuracy": "{:.2%}", "MCC": "{:+.4f}"}),
                         hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Bottom 10 Tickers (by MCC)**")
            st.dataframe(tick_df.tail(10)[["Ticker", "Accuracy", "MCC", "Samples"]]
                         .style.format({"Accuracy": "{:.2%}", "MCC": "{:+.4f}"}),
                         hide_index=True, use_container_width=True)


# -- What This Means -----------------------------------------------------------

elif page == "What This Means":
    st.title("What Do These Results Mean?")

    st.error("""
**All baselines are essentially random on the test set (MCC close to 0)**

- Best accuracy across all 12 experiments: ~52%
- Best MCC: ~+0.04 (Logistic Regression on FS3: Price + Tweets)

This means structured features alone (price technicals, fundamentals, tweet counts)
CANNOT reliably predict next-day stock movement -- regardless of the model complexity.
    """)

    st.divider()
    st.subheader("Why This Is Good for the Paper")
    st.markdown("""
These results directly justify the three novelty claims of MMGTFFF:

| Observation | Motivates |
|---|---|
| Price technicals alone -> random | Need richer features -> **FinBERT tweet embeddings (Phase 2)** |
| Fundamentals don't help much | Need structural context -> **Graph between stocks (Phase 4)** |
| Neural models (LSTM/MLP) can't beat LR | Need better architectures -> **Transformer encoder (Phase 2)** |
| Single global model -> random | Need sector-aware training -> **Federated Learning (Phase 5)** |

Even a **+2-3% MCC improvement** over these baselines with statistical significance is a publishable result.
    """)

    st.divider()
    st.subheader("Per-Sector Insight")
    st.markdown("""
From the per-sector breakdown:
- **Financial** sector shows the highest MCC (~0.15 for LR on FS3), suggesting tweet counts
  carry some signal for financial stocks
- **Basic Materials** is consistently near or below random
- This variance across sectors motivates **per-sector federated training** in Phase 5
    """)

    st.divider()
    st.subheader("Table 1 -- Ready for Paper")

    # Build paper table from actual results
    paper_rows = [
        {"Model": "Random Guess", "Feature Set": "-",
         "Accuracy": "50.00%", "F1": "0.5000", "MCC": "0.0000"},
    ]
    if full_results:
        for fs_key in FS_KEYS:
            for model in MODEL_NAMES:
                m = full_results.get(fs_key, {}).get('models', {}).get(model, {})
                if m:
                    paper_rows.append({
                        "Model": model,
                        "Feature Set": FS_SHORT[fs_key],
                        "Accuracy": f"{m.get('accuracy', 0):.2%}",
                        "F1": f"{m.get('f1', 0):.4f}",
                        "MCC": f"{m.get('mcc', 0):+.4f}",
                    })
    paper_rows.append({
        "Model": "MMGTFFF (Ours)", "Feature Set": "Full + Graph + Fed",
        "Accuracy": "TBD", "F1": "TBD", "MCC": "TBD",
    })
    st.table(pd.DataFrame(paper_rows))
    st.caption("Test set: ~4,921 samples across 87 tickers, 9 sectors. Per-ticker 80/20 chronological split.")
