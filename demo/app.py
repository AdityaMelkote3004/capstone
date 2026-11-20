"""Beyond Pooled Scores: interactive demo.

Replays the saved test predictions of the paper's runs (StockNet, Oct-Dec 2015,
64 trading days, 87 stocks). Nothing is retrained here; every number is computed
live from those predictions with the paper's definitions.

Run from the repository root:  streamlit run demo/app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from demo.metrics import (auc, counts, day_scores, logit, mcc_from, oracle_mcc,  # noqa: E402
                          scoreboard)

st.set_page_config(page_title="Beyond Pooled Scores: demo", layout="wide")

UP, DOWN, NEUTRAL, ACCENT = "#2b7a4b", "#b2443a", "#9aa5ab", "#1d6a86"
LABEL_NAME = {1: "Up", 0: "Down", -1: "Neutral band"}
LABEL_COLOUR = {"Up": UP, "Down": DOWN, "Neutral band": NEUTRAL}
CONTROLS = ["MMGT", "MMGT-D2", "Control: no zero-sum constraint",
            "Control: no cross-stock attention", "Control: wider single MMGT"]


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    d = pd.read_parquet(os.path.join(HERE, "demo_data.parquet"))
    for c in ("date", "model", "ticker", "sector"):
        d[c] = d[c].astype(str)
    return d


@st.cache_data
def board():
    """Scores per model. Read from demo/scoreboard.csv (written by build_demo_data.py from the
    same predictions) when present, so the first page loads quickly; otherwise computed."""
    path = os.path.join(HERE, "scoreboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col="Model")
    return scoreboard(load())


@st.cache_data
def featured_day():
    """A mixed day on which MMGT-D2's ranking is clearly visible: the highest Rank IC of its
    run-averaged predictions among days with at least 15 risers and 15 fallers."""
    d = load()
    d = d[d["model"] == "MMGT-D2"]
    avg = d.groupby(["date", "ticker"]).agg(prob=("prob", "mean"), ret=("ret", "first"),
                                            label=("label", "first")).reset_index()
    best, best_ic = None, -2.0
    for date, g in avg.groupby("date"):
        if (g["label"] == 1).sum() >= 15 and (g["label"] == 0).sum() >= 15:
            ic = g["prob"].rank().corr(g["ret"].rank())
            if ic > best_ic:
                best, best_ic = date, ic
    return best


@st.cache_data
def most_lopsided_day():
    d = load()
    d = d[(d["model"] == "MMGT") & (d["seed"] == d["seed"].min()) & (d["label"] >= 0)]
    share = d.groupby("date")["label"].mean()
    return share.sub(0.5).abs().idxmax(), share


@st.cache_data
def oracle():
    return oracle_mcc(load())


@st.cache_data
def ci_from_paper():
    """95% block-bootstrap intervals of Rank IC from the paper's analysis, if present."""
    path = os.path.join(os.path.dirname(HERE), "paper", "rev7_stats.json")
    if not os.path.exists(path):
        return {}
    s = json.load(open(path))
    names = {"D2-unconstrained": "Control: no zero-sum constraint",
             "D2-no-attention": "Control: no cross-stock attention",
             "MMGT-wide": "Control: wider single MMGT"}
    return {names.get(k, k): v["ic"] for k, v in s.get("level_ci", {}).items() if "ic" in v}


@st.cache_data
def blend_curve(model, alphas):
    """Pooled MCC (mean over runs) when each prediction is blended with its day's mean."""
    d = load()
    d = d[(d["model"] == model) & (d["label"] >= 0)].copy()
    d["daymean"] = d.groupby(["seed", "date"])["prob"].transform("mean")
    out = []
    for a in alphas:
        p = a * d["daymean"] + (1 - a) * d["prob"]
        vals = [mcc_from(counts(g["label"], (p[g.index] >= 0.5).astype(int)))
                for _, g in d.groupby("seed")]
        out.append(float(np.mean(vals)))
    return out


@st.cache_data
def day_table(date):
    d = load()
    d = d[d["date"] == date]
    rows = []
    for model, g in d.groupby("model", sort=False):
        runs = [day_scores(r) for _, r in g.groupby("seed")]
        market_up = np.nanmean(g[g["seed"] == g["seed"].min()]["ret"]) > 0
        calls = [((r["collapse"][0] + r["collapse"][2]) > 0) for r in runs]   # predicted "up" for the day
        rows.append({
            "Model": model,
            "Runs calling the day right": f"{np.mean([c == market_up for c in calls]):.0%}",
            "Within-day MCC": np.nanmean([r["wd"] for r in runs]),
            "Pairwise accuracy": np.nanmean([r["pairacc"] for r in runs]),
            "Rank IC": np.nanmean([r["ic"] for r in runs]),
        })
    return pd.DataFrame(rows).set_index("Model")


def nice_date(s):
    return pd.Timestamp(s).strftime("%a %d %b %Y")


data = load()
DATES = sorted(data["date"].unique())
MODELS = list(dict.fromkeys(data["model"]))

# ---------------------------------------------------------------- header
st.title("Beyond Pooled Scores")
st.markdown(
    "Can a stock model be good for the wrong reason? This demo replays the **saved test "
    "predictions** of our paper's models on StockNet (87 US stocks, 64 trading days, "
    "Oct to Dec 2015). Nothing is retrained; every number is computed from those predictions.")

with st.sidebar:
    st.header("About this demo")
    st.markdown(
        "- **Market timing**: predicting whether most stocks rise on a day.\n"
        "- **Stock selection**: predicting which stocks do better than others that day.\n"
        "- Pooled scores mix the two. Our measures separate them.")
    st.caption("PES University capstone PW26_MHR_03. Aditya M Hubli, Achintya N, "
               "Aditya Melkote, Aayush Prem. Guide: Dr. Mamatha H R.")

tabs = st.tabs(["1. The trap", "2. One trading day", "3. Scoreboard",
                "4. Separating market and stocks", "5. What we claim"])

# ---------------------------------------------------------------- 1. the trap
with tabs[0]:
    b = board()
    c1, c2, c3 = st.columns(3)
    c1.metric("Hindsight oracle (pooled MCC)", f"{oracle():.3f}",
              help="Knows only whether most stocks rose each day, and predicts that for every stock.")
    c2.metric("MMGT, our model (pooled MCC)", f"{b.loc['MMGT', 'Pooled MCC']:.3f}")
    base = b.loc[["Logistic regression", "MLP", "LSTM", "ALSTM"], "Pooled MCC"].max()
    c3.metric("Best tuned baseline (pooled MCC)", f"{base:.3f}")
    st.markdown(
        "A cheat that knows **nothing about any single stock**, only each day's majority "
        "direction, scores far above every real model. So the usual pooled score can be "
        "earned almost entirely by calling the market's direction.")

    st.subheader("Experiment: erase the differences between stocks")
    st.markdown(
        "Blend each prediction with the average prediction of its day. At 100% every stock "
        "gets the same prediction, so only the day-level call is left.")
    alpha = st.slider("Share of each prediction replaced by its day's average", 0, 100, 0, 10,
                      format="%d%%", key="alpha")
    alphas = [i / 10 for i in range(11)]
    fig = go.Figure()
    for model, colour in (("MMGT", ACCENT), ("LSTM", "#8a5a00")):
        ys = blend_curve(model, alphas)
        fig.add_trace(go.Scatter(x=[a * 100 for a in alphas], y=ys, mode="lines+markers",
                                 name=model, line=dict(color=colour, width=2)))
        fig.add_trace(go.Scatter(x=[alpha], y=[ys[alpha // 10]], mode="markers", showlegend=False,
                                 marker=dict(size=14, color=colour, line=dict(width=2, color="white"))))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Share replaced by the day's average (%)", yaxis_title="Pooled MCC",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig)
    mm = blend_curve("MMGT", alphas)
    st.info(f"MMGT scores {mm[0]:.3f} as it is and {mm[-1]:.3f} with every difference between "
            "stocks erased. Its score comes from calling the day, not from telling stocks apart. "
            "The per-stock LSTM behaves differently.")

# ---------------------------------------------------------------- 2. one day
with tabs[1]:
    one = data[(data["model"] == "MMGT") & (data["seed"] == data["seed"].min())]
    lop, share_up = most_lopsided_day()
    date = st.selectbox("Trading day", DATES, index=DATES.index(featured_day()),
                        format_func=nice_date, key="date")
    st.caption(f"Opens on a mixed day where MMGT-D2 ranks stocks well. For the trap, try "
               f"{nice_date(lop)}: {share_up[lop]:.0%} of labelled stocks moved the same way, so "
               "calling the day is enough to look good.")
    day = one[one["date"] == date]
    lab = day["label"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stocks that rose (labelled up)", int(lab.get(1, 0)))
    k2.metric("Stocks that fell (labelled down)", int(lab.get(0, 0)))
    k3.metric("Tiny moves (neutral band)", int(lab.get(-1, 0)))
    k4.metric("Average return", f"{day['ret'].mean():+.2f}%",
              help="Equal-weighted over the stocks with a full input window: the market's direction M_t.")

    model = st.selectbox("Model to inspect", MODELS, index=MODELS.index("MMGT-D2"), key="model")
    g = data[(data["model"] == model) & (data["date"] == date)]
    avg = g.groupby(["ticker", "sector"], observed=True).agg(
        prob=("prob", "mean"), ret=("ret", "first"), label=("label", "first"),
        m=("m", "mean"), e=("e", "mean")).reset_index()
    avg["Outcome"] = avg["label"].map(LABEL_NAME)

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for name, colour in LABEL_COLOUR.items():
            s = avg[avg["Outcome"] == name]
            fig.add_trace(go.Scatter(
                x=s["prob"], y=s["ret"], mode="markers", name=name,
                marker=dict(color=colour, size=9, line=dict(width=1, color="white")),
                text=s["ticker"] + " (" + s["sector"].astype(str) + ")",
                hovertemplate="%{text}<br>predicted %{x:.3f}<br>return %{y:+.2f}%<extra></extra>"))
        fig.add_vline(x=avg["prob"].mean(), line_dash="dash", line_color=ACCENT,
                      annotation_text="day's average prediction")
        fig.add_vline(x=0.5, line_color="#888", line_width=1)
        fig.add_hline(y=0, line_color="#888", line_width=1)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title=f"{model}: predicted probability of a rise (mean over runs)",
                          yaxis_title="Actual return that day (%)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
        st.caption("Market timing is where the cloud sits: right of 0.5 means the model calls an up "
                   "day. Stock selection is whether the cloud slopes upward: stocks predicted higher "
                   "should really do better.")
    with right:
        st.markdown("**Every model on this day** (averaged over its runs)")
        t = day_table(date)
        st.dataframe(t.style.format({"Within-day MCC": "{:+.3f}", "Pairwise accuracy": "{:.3f}",
                                     "Rank IC": "{:+.3f}"}, na_rep="n/a"),
                     width="stretch", height=460)

    if avg["e"].notna().any():
        st.subheader(f"{model}: market part and stock parts on this day")
        ordered = avg.sort_values("e", ascending=False)
        fig = go.Figure(go.Bar(x=ordered["ticker"], y=ordered["e"],
                               marker_color=[LABEL_COLOUR[o] for o in ordered["Outcome"]],
                               hovertemplate="%{x}<br>excess logit %{y:+.3f}<extra></extra>"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Stock part e (excess logit)", xaxis_tickangle=-90)
        st.plotly_chart(fig)
        a1, a2, a3 = st.columns(3)
        a1.metric("Market part m (one number for the day)", f"{avg['m'].mean():+.3f}")
        a2.metric("Sum of the stock parts", f"{avg['e'].sum():+.4f}")
        zbar = float(np.mean([logit(r["prob"]).mean() for _, r in g.groupby("seed")]))
        a3.metric("Average logit of the day", f"{zbar:+.3f}",
                  help="Mean over runs of the day's average logit; equals the market part when the "
                       "stock parts sum to zero.")
        if abs(avg["e"].sum()) < 1e-3:
            st.success("The stock parts add up to zero, so they can only rank stocks against each "
                       "other. The day-level call comes from the market part alone.")
        else:
            st.warning("Here the stock parts do not add up to zero: without the constraint, the "
                       "stock branch also shifts the whole day.")

# ---------------------------------------------------------------- 3. scoreboard
with tabs[2]:
    b = board()
    st.markdown("Mean over training runs, computed from the saved predictions with the paper's "
                "definitions. These reproduce Table I and the controls of the paper.")
    show = b.drop(columns=["MCC s.d."])
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "Runs"}),
                 width="stretch", height=470)
    with st.expander("What each column measures"):
        st.markdown(
            "- **Pooled MCC**: the usual score over all labelled stock-days (threshold 0.5).\n"
            "- **Within-day MCC**: split each day's labelled stocks at the day's median prediction; "
            "average over days. Moving a whole day up or down cannot change it.\n"
            "- **Pairwise accuracy**: share of each day's (rose, fell) pairs ordered correctly; "
            "the daily AUC. 0.5 is chance.\n"
            "- **Rank IC**: daily rank correlation between predictions and actual returns over "
            "all stocks, including tiny moves.\n"
            "- **Day-collapse MCC**: replace every prediction by its day's average probability.\n"
            "- **Market-logit MCC** and **Daily AUC**: use the day's average logit, which for "
            "MMGT-D and MMGT-D2 is exactly their market part.")
    st.subheader("Market timing against stock selection")
    cx, cy = st.columns(2)
    xm = cx.selectbox("Market timing measure", ["Day-collapse MCC", "Market-logit MCC", "Daily AUC"],
                      key="xm")
    ym = cy.selectbox("Stock selection measure", ["Within-day MCC", "Pairwise accuracy", "Rank IC"],
                      index=2, key="ym")
    colours = [ACCENT if m.startswith("MMGT") else ("#8a5a00" if m.startswith("Control") else "#7d878d")
               for m in b.index]
    fig = go.Figure(go.Scatter(x=b[xm], y=b[ym], mode="markers+text", text=b.index,
                               textposition="top center", textfont=dict(size=11),
                               marker=dict(size=12, color=colours, line=dict(width=1, color="white")),
                               hovertemplate="%{text}<br>%{x:.3f}, %{y:.3f}<extra></extra>"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=xm, yaxis_title=ym)
    st.plotly_chart(fig)
    st.caption("Right means better market timing; up means better stock selection. Grey: "
               "baselines; blue: our models; brown: the final controls.")

# ---------------------------------------------------------------- 4. separation
with tabs[3]:
    st.markdown(
        "MMGT-D2 writes every prediction as **market part + stock part**: "
        "z(stock, day) = m(day) + e(stock, day). The stock parts are forced to **sum to zero** "
        "each day, so they can only rank stocks. The market part comes from a full MMGT.")
    st.subheader("Try it: the zero-sum trick")
    m = st.slider("Market part m for the day", -2.0, 2.0, 0.4, 0.1, key="m")
    cols = st.columns(4)
    raw = [cols[i].slider(f"Stock {n}: raw score", -2.0, 2.0, v, 0.1, key=f"raw{n}")
           for i, (n, v) in enumerate(zip("ABCD", (1.0, 0.6, 0.1, -0.1)))]
    e = np.array(raw) - np.mean(raw)
    z = m + e
    p = 1 / (1 + np.exp(-z))
    st.dataframe(pd.DataFrame({"Stock": list("ABCD"), "Stock part e (centred)": e,
                               "Logit z = m + e": z, "Probability": p}).set_index("Stock")
                 .style.format("{:+.3f}"), width="stretch")
    s1, s2, s3 = st.columns(3)
    s1.metric("Average logit", f"{z.mean():+.3f}", help="Always equal to m.")
    s2.metric("Average probability", f"{p.mean():.3f}")
    s3.metric("Probability of m itself", f"{1 / (1 + np.exp(-m)):.3f}")
    st.caption("The average logit is exactly m whatever the stock scores are. The average "
               "probability is close to the probability of m but not equal, because the sigmoid is "
               "curved. That is why the paper scores the market part directly (market-logit MCC).")

    st.subheader("Why it works: the final controls")
    b = board()
    cis = ci_from_paper()
    vals = b.loc[CONTROLS, "Rank IC"]
    err_hi = [cis[c][1] - vals[c] if c in cis else 0 for c in CONTROLS]
    err_lo = [vals[c] - cis[c][0] if c in cis else 0 for c in CONTROLS]
    fig = go.Figure(go.Bar(
        x=CONTROLS, y=vals, marker_color=[NEUTRAL, ACCENT, "#8a5a00", "#8a5a00", "#8a5a00"],
        error_y=dict(type="data", array=err_hi, arrayminus=err_lo, color="#666") if cis else None,
        hovertemplate="%{x}<br>Rank IC %{y:.3f}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Rank IC (mean of 10 runs)")
    st.plotly_chart(fig)
    st.markdown(
        "- **Remove the zero-sum constraint** (same two branches, same ranking loss): Rank IC falls "
        "back to MMGT's level. The constraint contributes.\n"
        "- **Same size as one MMGT**: no gain either, so extra capacity alone does not explain it.\n"
        "- **No cross-stock attention in the stock branch**: in between; not resolved.")
    st.caption("Error bars: 95% intervals from the paper's hierarchical block bootstrap, when "
               "paper/rev7_stats.json is present.")

# ---------------------------------------------------------------- 5. claims
with tabs[4]:
    a, b2 = st.columns(2)
    with a:
        st.subheader("What we show")
        st.markdown(
            "- Pooled scores on StockNet can be earned mostly by **market timing** (oracle 0.625).\n"
            "- MMGT beats our tuned baselines on the pooled score, but mainly by timing the "
            "market; its Rank IC is 0.001.\n"
            "- MMGT-D2 improves full-universe Rank IC to 0.027 on the main test period, with no "
            "detectable change in pooled MCC.\n"
            "- Controls: the zero-sum constraint contributes; extra capacity alone does not.")
    with b2:
        st.subheader("What we do not claim")
        st.markdown(
            "- A new state of the art (our DTML backbone reproduction: 0.139 vs 0.191 published).\n"
            "- Better up/down discrimination by MMGT-D2 (pairwise accuracy +0.004, p = 0.67).\n"
            "- That the gain generalises: it does not recur in June to July 2015, and StockNet has "
            "no later data.\n"
            "- That MMGT-D2 is disentangled: its stock branch still encodes past market moves.\n"
            "- Any trading profit.")
    st.caption("All follow-up experiments were pre-registered before they were run; see "
               "results/experiment11_dtml/PREREGISTRATION.md.")
