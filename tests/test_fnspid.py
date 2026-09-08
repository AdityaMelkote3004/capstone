import pandas as pd

from src.data.fnspid import (classify_row, coverage_report, is_english,
                             stream_filter)

UNIVERSE = {"AAPL", "MSFT", "BRK-A"}

HEADER = ("Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,"
          "Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary\n")


def _row(ts: str, title: str, symbol: str = "") -> str:
    return f"{ts},{title},{symbol},http://u,Pub,Auth,,,,,\n"


def test_is_english_accepts_ascii_headlines():
    assert is_english("Market Wrap For April 23: S&P 500 Ends Winning Streak")


def test_is_english_rejects_cyrillic():
    assert not is_english("Бюджет-2005 принят Госдумой в третьем чтении")


def test_is_english_handles_empty_and_none():
    assert not is_english("")
    assert not is_english(None)


def test_classify_row_splits_macro_from_micro():
    assert classify_row("", UNIVERSE) == "macro"
    assert classify_row("AAPL", UNIVERSE) == "micro"
    assert classify_row("TSLA", UNIVERSE) == "skip", "outside the universe"


def test_stream_filter_keeps_only_the_window():
    lines = [HEADER,
             _row("2013-12-31 10:00:00 UTC", "too early"),
             _row("2014-06-02 10:00:00 UTC", "in window"),
             _row("2016-01-04 10:00:00 UTC", "too late")]
    macro, micro = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["in window"]
    assert micro == []


def test_stream_filter_routes_to_both_channels():
    lines = [HEADER,
             _row("2014-06-02 10:00:00 UTC", "market wide"),
             _row("2014-06-02 11:00:00 UTC", "apple news", "AAPL"),
             _row("2014-06-02 12:00:00 UTC", "other co", "TSLA")]
    macro, micro = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["market wide"]
    assert [r["headline"] for r in micro] == ["apple news"]
    assert micro[0]["symbol"] == "AAPL"


def test_stream_filter_drops_non_english():
    lines = [HEADER,
             _row("2014-06-02 10:00:00 UTC", "Бюджет принят Госдумой"),
             _row("2014-06-02 11:00:00 UTC", "normal headline")]
    macro, _ = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["normal headline"]


def test_stream_filter_survives_malformed_lines():
    """A 5.34 GB file will contain broken rows; one must not abort the pass."""
    lines = [HEADER,
             "garbage with no comma structure\n",
             ",,,,,,,,,,\n",
             _row("2014-06-02 10:00:00 UTC", "good row")]
    macro, _ = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["good row"]


def test_stream_filter_preserves_time_of_day():
    """Leakage control depends on the timestamp, not just the date."""
    lines = [HEADER, _row("2014-06-02 13:45:07 UTC", "timed")]
    macro, _ = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert macro[0]["timestamp"] == pd.Timestamp("2014-06-02 13:45:07")


def test_coverage_report_counts_days_and_flags_the_gate():
    days = [f"2014-01-{d:02d}" for d in range(1, 11)]
    rows = [{"timestamp": pd.Timestamp(f"{d} 10:00:00"), "headline": "h", "symbol": ""}
            for d in days[:6]]
    report = coverage_report(rows, days)
    assert report["days_covered"] == 6
    assert report["days_total"] == 10
    assert report["coverage_fraction"] == 0.6
    assert report["passes_gate"] is False, "0.6 is below the 0.70 gate"


def test_unterminated_quote_does_not_swallow_following_rows():
    """A single malformed quote must cost at most its own row, never the rows after it."""
    lines = [HEADER,
             '2014-06-02 10:00:00 UTC,"unterminated,AAPL,http://u,Pub,Auth,,,,,\n',
             _row("2014-06-03 10:00:00 UTC", "good one", "AAPL"),
             _row("2014-06-04 10:00:00 UTC", "good two", "MSFT")]
    _, micro = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in micro] == ["good one", "good two"]


def test_multiline_article_continuation_lines_are_skipped():
    """Article bodies span lines; continuation lines have no leading timestamp and
    must not be mistaken for rows."""
    lines = [HEADER,
             _row("2014-06-02 10:00:00 UTC", "real headline", "AAPL"),
             "this is a continuation line of an article body, with commas\n",
             "and another one\n",
             _row("2014-06-03 10:00:00 UTC", "second headline", "MSFT")]
    _, micro = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in micro] == ["real headline", "second headline"]


def test_quoted_headline_with_commas_still_parses():
    """Regression: the per-line parse must still honour quoting within the line."""
    lines = [HEADER,
             '2014-06-02 10:00:00 UTC,"B of A Maintains Neutral, Raises Target",AAPL,http://u,Pub,Auth,,,,,\n']
    _, micro = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert micro[0]["headline"] == "B of A Maintains Neutral, Raises Target"
    assert micro[0]["symbol"] == "AAPL"
