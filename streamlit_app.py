"""
Streamlit Dashboard — LR Baseline
MMGTFFF Capstone | Phase 1 | Member 1 (Aayush Prem)

Displays results from the stepwise modality baseline:
  Step 1 — Price Only
  Step 2 — Price + Sentiment
  Step 3 — Price + Fundamentals
  Step 4 — Full Structured

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LR Baseline Dashboard — MMGTFFF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS (matches teammate's style) ─────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #000000; }
    .stMetric {
        background-color: #1C1C1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #48484A;
    }
    h1, h2, h3 { color: #FFFFFF !important; }
    .stSelectbox label { color: #FFFFFF !important; }
    .stButton button {
        background-color: #0A84FF;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
    }
    .stButton button:hover { background-color: #0066CC; }
    .step-card {
        background-color: #1C1C1E;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #48484A;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/phase1_baselines/logistic_regression")
STEP_NAMES  = [
    "Step 1 — Price Only",
    "Step 2 — Price + Sentiment",
    "Step 3 — Price + Fundamentals",
    "Step 4 — Full Structured",
]
STEP_KEYS   = ["price", "price_sentiment", "price_fund", "full"]
STEP_SLUGS  = [
    "step1_price_only",
    "step2_price_sentiment",
    "step3_price_fundamentals",
    "step4_full_structured",
]
STEP_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
SECTOR_ORDER = [
    "Technology", "Financial", "Healthcare", "Consumer_Goods",
    "Services", "Utilities", "Conglomerates", "Basic_Materials",
    "Industrial_Goods",
]


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_metrics():
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_summary():
    path = RESULTS_DIR / "summary_table.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_per_sector(slug):
    path = RESULTS_DIR / "per_sector" / f"{slug}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_raw_data(csv_path):
    """Load and preprocess the full dataset for live exploration."""
    from data_loader import load_data
    return load_data(csv_path)


def results_exist():
    return (RESULTS_DIR / "metrics.json").exists()


def run_pipeline(csv_path):
    """Run the full baseline pipeline and stream output to Streamlit."""
    import subprocess
    cmd = [
        sys.executable, "run_baseline.py",
        "--data", csv_path,
        "--out", str(RESULTS_DIR),
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(Path(__file__).parent),
    )
    log_box = st.empty()
    logs = []
    for line in process.stdout:
        logs.append(line.rstrip())
        log_box.code("\n".join(logs[-30:]))   # show last 30 lines
    process.wait()
    return process.returncode == 0


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 LR Baseline")
    st.caption("MMGTFFF — Phase 1 | Aayush Prem")
    st.markdown("---")

    st.markdown("### ⚙️ Run Pipeline")
    csv_path = st.text_input(
        "CSV path",
        value="stocknet_final_modeling_set.csv",
        help="Path to stocknet_final_modeling_set.csv",
    )
    run_btn = st.button("▶ Run Baseline", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📅 Split Used")
    st.markdown("""
    | Split | Range |
    |-------|-------|
    | Train | Jan 2014 – Mar 2015 |
    | Val   | Apr 2015 – Jul 2015 |
    | Test  | Aug 2015 – Dec 2015 |
    """)

    st.markdown("---")
    st.markdown("### ℹ️ Leakage Fix")
    st.info(
        "`Return` = `sign(Return)` = `Target` (100% correlation). "
        "All same-day technical features are **lagged by 1 trading day** per ticker."
    )

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Step-by-Step", "Sector Analysis", "Feature Importance", "Raw Data"],
        index=0,
    )


# ── Run pipeline if requested ──────────────────────────────────────────────────

if run_btn:
    with st.spinner("Running baseline pipeline..."):
        success = run_pipeline(csv_path)
    if success:
        st.success("✅ Pipeline complete! Results loaded below.")
        st.cache_data.clear()
    else:
        st.error("❌ Pipeline failed. Check the log above.")

if not results_exist():
    st.warning(
        "No results found yet. Enter your CSV path in the sidebar and click **▶ Run Baseline**."
    )
    st.stop()

# ── Load results ──────────────────────────────────────────────────────────────
metrics_data = load_metrics()
summary_df   = load_summary()
steps        = metrics_data["steps"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("📊 LR Baseline — Overview")
    st.caption("Stepwise modality addition | Logistic Regression | Test: Aug–Dec 2015")
    st.markdown("---")

    # ── Top metric cards ──────────────────────────────────────────────────────
    cols = st.columns(4)
    for col, step, colour in zip(cols, steps, STEP_COLOURS):
        with col:
            label = step["name"].split("—")[1].strip()
            st.markdown(
                f"<div class='step-card'>"
                f"<b style='color:{colour}'>{label}</b><br>"
                f"<span style='font-size:28px;font-weight:bold'>{step['auc']:.4f}</span><br>"
                f"<span style='color:#8E8E93'>AUC-ROC</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("📋 Summary Table")
    display_df = summary_df.copy()
    display_df.index = [f"S{i+1}" for i in range(len(display_df))]
    st.dataframe(
        display_df.style
            .format({"Accuracy": "{:.4f}", "F1_Macro": "{:.4f}", "MCC": "{:.4f}", "AUC_ROC": "{:.4f}"})
            .background_gradient(subset=["AUC_ROC"], cmap="Greens"),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.subheader("📈 Metrics Across Steps")
    metric_keys   = ["accuracy", "f1_macro", "mcc", "auc"]
    metric_labels = ["Accuracy", "F1 (macro)", "MCC", "AUC-ROC"]
    short_names   = ["Price Only", "Price+Sent", "Price+Fund", "Full"]

    fig = go.Figure()
    for mk, ml in zip(metric_keys, metric_labels):
        fig.add_trace(go.Bar(
            name=ml,
            x=short_names,
            y=[s[mk] for s in steps],
            text=[f"{s[mk]:.3f}" for s in steps],
            textposition="outside",
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="Random baseline (0.5)")
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#1C1C1E",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0.35, 0.65]),
        margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── AUC delta annotations ─────────────────────────────────────────────────
    st.markdown("#### 🔍 Incremental AUC Gain per Modality")
    d_cols = st.columns(3)
    deltas = [
        ("+ Sentiment counts", steps[1]["auc"] - steps[0]["auc"]),
        ("+ Fundamentals",     steps[2]["auc"] - steps[0]["auc"]),
        ("+ Both together",    steps[3]["auc"] - steps[0]["auc"]),
    ]
    for col, (label, delta) in zip(d_cols, deltas):
        col.metric(label, f"{delta:+.4f} AUC", delta_color="normal" if delta > 0 else "inverse")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STEP-BY-STEP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Step-by-Step":
    st.title("🔢 Step-by-Step Results")
    st.caption("Each step adds one modality on top of the previous")
    st.markdown("---")

    selected_step = st.selectbox("Select Step", STEP_NAMES)
    idx  = STEP_NAMES.index(selected_step)
    step = steps[idx]

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",   f"{step['accuracy']:.4f}")
    c2.metric("F1 (macro)", f"{step['f1_macro']:.4f}")
    c3.metric("MCC",        f"{step['mcc']:.4f}")

    auc_delta = step["auc"] - steps[0]["auc"] if idx > 0 else None
    c4.metric(
        "AUC-ROC", f"{step['auc']:.4f}",
        delta=f"{auc_delta:+.4f} vs Price Only" if auc_delta is not None else None,
    )

    st.markdown("---")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrix")
    cm = np.array(step["confusion_matrix"])
    cm_pct = cm / cm.sum() * 100

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred ↓ (Down)", "Pred ↑ (Up)"],
        y=["True ↓ (Down)", "True ↑ (Up)"],
        text=[[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#1C1C1E",
        height=320,
        margin=dict(t=20),
    )
    col_cm, col_stats = st.columns([1, 1])
    with col_cm:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_stats:
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        total = cm.sum()
        st.markdown("**Derived stats:**")
        st.markdown(f"- True Negative Rate (Specificity): `{tn/(tn+fp):.3f}`")
        st.markdown(f"- True Positive Rate (Recall):      `{tp/(tp+fn):.3f}`")
        st.markdown(f"- Precision (Up class):             `{tp/(tp+fp):.3f}`")
        st.markdown(f"- Total test samples:               `{total}`")
        st.markdown(f"- Correctly classified:             `{tn+tp}` / `{total}`")

    st.markdown("---")

    # ── Feature description ───────────────────────────────────────────────────
    st.subheader("📦 Features Used")
    desc = {
        0: "**15 lagged technical indicators** — RSI, MACD, MACD_Signal, MACD_Hist, "
           "Volatility_5/20, MA_5/10/20, Price_MA_Ratios, Return, Volume_Change, HL_Spread. "
           "All shifted 1 trading day to prevent leakage.",
        1: "Price Only features **+** 3 tweet-count columns — "
           "`Company_Tweet_Count`, `Event_Tweet_Count`, `Total_Tweet_Count` (also lagged by 1 day). "
           "No NLP — just volume of social media activity.",
        2: "Price Only features **+** all 14 EDGAR fundamentals — Revenue, NetIncome, EPS, "
           "Profit_Margin, Debt_To_Equity, ROA, Current_Ratio, Operating_Margin, etc. "
           "Each fundamental gets a binary presence mask (28 cols total). "
           "Forward-filled within ticker, zero-filled otherwise.",
        3: "**All modalities combined** — Price (15) + Sentiment (3) + Fundamentals (28) = 46 features. "
           "This is the upper bound for any non-deep-learning model on this data.",
    }
    st.info(desc[idx])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SECTOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Sector Analysis":
    st.title("🏭 Per-Sector Analysis")
    st.caption("AUC-ROC broken down by sector for each step")
    st.markdown("---")

    # Build combined sector table
    sector_frames = []
    for i, (slug, sname) in enumerate(zip(STEP_SLUGS, ["S1","S2","S3","S4"])):
        df_s = load_per_sector(slug)
        if df_s is not None:
            df_s = df_s.rename(columns={"AUC": sname})
            sector_frames.append(df_s.set_index("Sector")[[sname]])

    if sector_frames:
        heatmap_df = pd.concat(sector_frames, axis=1).reset_index()

        # ── Heatmap ──────────────────────────────────────────────────────────
        st.subheader("🌡️ AUC Heatmap — Sector × Step")
        z_vals  = heatmap_df[["S1","S2","S3","S4"]].values
        sectors = heatmap_df["Sector"].tolist()

        fig_heat = go.Figure(go.Heatmap(
            z=z_vals,
            x=["S1: Price", "S2: +Sentiment", "S3: +Fundamentals", "S4: Full"],
            y=sectors,
            text=[[f"{v:.3f}" for v in row] for row in z_vals],
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0.44, zmax=0.60,
            colorbar=dict(title="AUC"),
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#000000",
            plot_bgcolor="#1C1C1E",
            height=420,
            margin=dict(t=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # ── Per-sector line chart ─────────────────────────────────────────────
        st.subheader("📈 AUC Trend by Sector Across Steps")
        selected_sectors = st.multiselect(
            "Select sectors to compare",
            options=sectors,
            default=sectors[:4],
        )

        fig_line = go.Figure()
        step_labels = ["S1: Price", "S2: +Sentiment", "S3: +Fundamentals", "S4: Full"]
        for sector in selected_sectors:
            row = heatmap_df[heatmap_df["Sector"] == sector].iloc[0]
            fig_line.add_trace(go.Scatter(
                x=step_labels,
                y=[row["S1"], row["S2"], row["S3"], row["S4"]],
                mode="lines+markers",
                name=sector,
                line=dict(width=2),
                marker=dict(size=8),
            ))

        fig_line.add_hline(y=0.5, line_dash="dash", line_color="grey",
                           annotation_text="Random (0.5)")
        fig_line.update_layout(
            template="plotly_dark",
            paper_bgcolor="#000000",
            plot_bgcolor="#1C1C1E",
            height=380,
            yaxis=dict(range=[0.43, 0.62], title="AUC-ROC"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        # ── Best modality per sector ──────────────────────────────────────────
        st.subheader("🏆 Best Step per Sector")
        best_rows = []
        for _, row in heatmap_df.iterrows():
            vals  = {"S1": row["S1"], "S2": row["S2"], "S3": row["S3"], "S4": row["S4"]}
            best  = max(vals, key=vals.get)
            label = {"S1": "Price Only", "S2": "+Sentiment", "S3": "+Fundamentals", "S4": "Full"}[best]
            best_rows.append({"Sector": row["Sector"], "Best Step": label, "Best AUC": round(vals[best], 4)})
        st.dataframe(pd.DataFrame(best_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Feature Importance":
    st.title("🔍 Feature Importance")
    st.caption("LR coefficients from the Full Structured model (Step 4) — standardised features")
    st.markdown("---")

    st.info(
        "Since all features are z-score normalised before training, LR coefficients are "
        "directly comparable in magnitude. A positive coefficient pushes the model toward "
        "predicting UP (1); negative pushes toward DOWN (0)."
    )

    # Re-run model to extract coefficients live
    if st.button("📥 Load Feature Coefficients"):
        with st.spinner("Loading model and extracting coefficients..."):
            try:
                from data_loader import load_data, get_feature_groups, get_xy
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler

                df = load_raw_data(csv_path)
                fg = get_feature_groups(df)
                features = fg["full"]

                scaler = StandardScaler()
                X_train, y_train, scaler = get_xy(df, features, "train", scaler, fit_scaler=True)
                X_test,  y_test,  _      = get_xy(df, features, "test",  scaler, fit_scaler=False)

                model = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", random_state=42)
                model.fit(X_train, y_train)

                coefs   = model.coef_[0]
                top_idx = np.argsort(np.abs(coefs))[::-1][:25]
                top_f   = [features[i].replace("_lag1", " (lag1)").replace("_present", " mask")
                           for i in top_idx]
                top_c   = coefs[top_idx]

                colours = ["#C44E52" if c > 0 else "#4C72B0" for c in top_c]

                fig_imp = go.Figure(go.Bar(
                    x=top_c,
                    y=top_f,
                    orientation="h",
                    marker_color=colours,
                    text=[f"{c:+.4f}" for c in top_c],
                    textposition="outside",
                ))
                fig_imp.add_vline(x=0, line_color="white", line_width=1)
                fig_imp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#000000",
                    plot_bgcolor="#1C1C1E",
                    height=700,
                    xaxis_title="LR Coefficient (standardised)",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=220, t=20),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("**Red** = pushes toward predicting UP &nbsp;|&nbsp; **Blue** = pushes toward DOWN")

            except Exception as e:
                st.error(f"Error: {e}. Make sure the CSV path is set correctly in the sidebar.")
    else:
        st.markdown("Click the button above to extract and plot feature coefficients live from the model.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RAW DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Raw Data":
    st.title("🗂️ Raw Data Explorer")
    st.caption("Explore the preprocessed StockNet dataset")
    st.markdown("---")

    if st.button("📂 Load Dataset"):
        with st.spinner("Loading dataset..."):
            try:
                df = load_raw_data(csv_path)
                st.session_state["df"] = df
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error: {e}")

    if "df" in st.session_state:
        df = st.session_state["df"]

        # ── Summary stats ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",  f"{len(df):,}")
        c2.metric("Tickers",     df["Ticker"].nunique())
        c3.metric("Sectors",     df["Sector"].nunique())
        c4.metric("Date Range",  f"{df['Date'].min().date()} → {df['Date'].max().date()}")

        st.markdown("---")

        # ── Filter by sector / ticker ─────────────────────────────────────────
        col_f1, col_f2 = st.columns(2)
        sector   = col_f1.selectbox("Filter by Sector", ["All"] + sorted(df["Sector"].unique()))
        filtered = df if sector == "All" else df[df["Sector"] == sector]
        ticker   = col_f2.selectbox("Filter by Ticker", ["All"] + sorted(filtered["Ticker"].unique()))
        filtered = filtered if ticker == "All" else filtered[filtered["Ticker"] == ticker]

        st.dataframe(
            filtered[["Date","Ticker","Sector","Close","Return","Target",
                       "RSI_14","MACD","Volume_Change","Total_Tweet_Count"]].head(200),
            use_container_width=True,
        )

        st.markdown("---")

        # ── Class balance per sector ───────────────────────────────────────────
        st.subheader("📊 Class Balance per Sector")
        bal = (df.groupby("Sector")["Target"].mean() * 100).reset_index()
        bal.columns = ["Sector", "Pct_Up"]
        fig_bal = px.bar(
            bal.sort_values("Pct_Up"),
            x="Pct_Up", y="Sector", orientation="h",
            color="Pct_Up",
            color_continuous_scale="RdYlGn",
            range_color=[45, 55],
            labels={"Pct_Up": "% Days Predicted Up"},
            template="plotly_dark",
        )
        fig_bal.add_vline(x=50, line_dash="dash", line_color="white",
                          annotation_text="50% balanced")
        fig_bal.update_layout(
            paper_bgcolor="#000000",
            plot_bgcolor="#1C1C1E",
            height=360,
            coloraxis_showscale=False,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_bal, use_container_width=True)
    else:
        st.markdown("Click **Load Dataset** above to explore the data.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#48484A'>"
    "MMGTFFF — Phase 1 LR Baseline | Aayush Prem | feat/data-baselines"
    "</div>",
    unsafe_allow_html=True,
)
