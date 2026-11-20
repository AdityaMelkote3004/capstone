# Demo: Beyond Pooled Scores

An interactive replay of the paper's saved test predictions (StockNet, 87 stocks,
64 trading days, Oct to Dec 2015). Nothing is trained live, so it starts in seconds
and works offline.

## Run it

```bash
pip install streamlit plotly pandas numpy scipy pyarrow
streamlit run demo/app.py          # from the repository root
```

It opens in the browser at http://localhost:8501. The first load takes a few seconds.

`demo/demo_data.parquet` (4 MB) holds the predictions, so the 178 files in
`results/experiment11_dtml/preds/` are not needed. To rebuild it from those files:
`python demo/build_demo_data.py`. It prints a scoreboard that must match Table I.

## Five-minute walkthrough

**1. The trap (about 1 min).** Start here.
- Point at the three numbers. A cheat that only knows each day's majority direction
  scores 0.625. MMGT scores 0.146, and the best baseline 0.057.
- Say: "The usual score can be earned by calling the market, without knowing anything
  about individual stocks."
- Drag the slider to 100%. MMGT's score *rises* from 0.146 to 0.219 when every
  difference between stocks is erased. So its score comes from calling the day.

**2. One trading day (about 1.5 min).**
- It opens on a mixed day. In the scatter, left/right is market timing and the upward
  slope is stock selection.
- With MMGT-D2 selected, scroll to the bar chart: one market number for the day, plus 87
  stock parts that add up to exactly 0.000.
- Switch the model to "Control: no zero-sum constraint". The sum is no longer zero, and
  a warning appears.
- Optional: pick the day named in the caption, where every labelled stock moved the same
  way. Within-day MCC and pairwise accuracy show "n/a" there, because no stock fell.

**3. Scoreboard (about 1 min).**
- This reproduces Table I and the controls from the saved predictions.
- In the scatter, choose "Rank IC" for stock selection. MMGT sits far right (timing) but
  low (selection); MMGT-D2 keeps most of the timing and moves up.

**4. Separating market and stocks (about 1 min).**
- Move the sliders in the zero-sum calculator. The average logit always equals m; the
  average probability does not, because the sigmoid is curved. That is why the paper
  scores the market part directly.
- The controls chart: without the constraint, or with a single model of the same size,
  Rank IC drops back to MMGT's level.

**5. What we claim (30 s).** End here. Read the "do not claim" list aloud: no state of
the art, no better up/down discrimination, not replicated in an earlier period, not
disentangled, no trading claims. Examiners respond well to this.

## If asked "is this live prediction?"

No. It replays the test predictions saved by every training run in the paper, and
computes the paper's metrics from them with the same definitions. Retraining takes 8 to
20 minutes per run on a CPU, which is too slow for a live demo.
