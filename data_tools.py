"""
data_tools.py
=============
XAUUSD M1 data handling:
  - load 3 tab-separated CSVs (<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>)
  - merge, parse datetime, sort, dedup
  - synthetic M1 gold generator (so the system can be TESTED end-to-end with honest numbers
    when the user's real CSVs are not yet present).

Drop real files into ./data/ named:
  GOLD_2023_2024.csv  GOLD_2024_2025.csv  GOLD_2025_2026.csv
and load_data() picks them up automatically.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd

DEFAULT_FILES = ["GOLD_2023_2024.csv", "GOLD_2024_2025.csv", "GOLD_2025_2026.csv"]
COLS = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]

# MT/MT5 exports store SPREAD in POINTS. Gold (2-decimal) -> 1 point = $0.01.
SPREAD_POINT = float(os.environ.get("SPREAD_POINT", "0.01"))


def _read_one(path: str) -> pd.DataFrame:
    # Header may or may not exist. If first row looks like '<DATE>', skip it.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
    skiprows = 1 if first.lower().startswith("<date>") else 0
    df = pd.read_csv(
        path, sep="\t", header=None, skiprows=skiprows, names=COLS,
        dtype=str, engine="python", on_bad_lines="skip",
    )
    return df


def load_data(data_dir: str = "data", files=None) -> pd.DataFrame | None:
    if files:
        paths = [os.path.join(data_dir, f) for f in files if os.path.exists(os.path.join(data_dir, f))]
    else:
        # glob ANY *.csv in the data dir so naming (e.g. "GOLD.i#_M1 2025 to 2026.csv") just works
        paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        return None

    parts = []
    for p in paths:
        try:
            parts.append(_read_one(p))
        except Exception as e:  # pragma: no cover
            print(f"[data_tools] WARN could not read {p}: {e}")
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)

    # MT dates look like "2025.06.24" -> normalise to "2025-06-24"
    dstr = df["date"].astype(str).str.replace(".", "-", regex=False)
    df["datetime"] = pd.to_datetime(dstr + " " + df["time"].astype(str), errors="coerce")
    df = df.dropna(subset=["datetime"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=("open", "high", "low", "close"))
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)

    df["vol"] = pd.to_numeric(df.get("tickvol", 0), errors="coerce").fillna(0.0)
    # SPREAD column is in POINTS -> convert to dollars (gold 2-decimal)
    sp = pd.to_numeric(df.get("spread", 0), errors="coerce").fillna(0.0)
    df["spread"] = sp * SPREAD_POINT
    return df[["datetime", "open", "high", "low", "close", "vol", "spread"]].copy()


def make_synthetic(n_bars: int = 60_000, start_price: float = 2000.0,
                   seed: int = 42, start_dt: str = "2024-01-01") -> pd.DataFrame:
    """
    Synthetic but gold-like M1 walk:
      - base vol ~$0.55/bar, fat tails, slow regime drift, weekend skip.
    Used ONLY for testing. NOT a model of real price.
    """
    rng = np.random.default_rng(seed)
    # regime drift: slowly varying bias
    regime = np.cumsum(rng.standard_normal(n_bars)) / 250.0
    regime = np.clip(regime, -1.0, 1.0)
    # per-bar shock: normal + occasional jump (fat tails)
    shock = rng.standard_normal(n_bars) * 0.45
    jump = (rng.random(n_bars) < 0.0008) * rng.standard_normal(n_bars) * 6.0
    step = regime * 0.012 + shock + jump

    close = start_price + np.cumsum(step)
    close = np.maximum(close, 100.0)

    # build OHLC from close with small intrabar range
    intrabar = np.abs(rng.standard_normal(n_bars)) * 0.30 + 0.05
    op = np.empty(n_bars); op[0] = start_price; op[1:] = close[:-1]
    hi = np.maximum(op, close) + intrabar * rng.random(n_bars)
    lo = np.minimum(op, close) - intrabar * rng.random(n_bars)

    # timestamps: 1-min, skip Sat/Sun (gold ~ closed weekends)
    ts = pd.date_range(start_dt, periods=n_bars * 2, freq="1min")
    ts = ts[ts.dayofweek < 5][:n_bars]

    df = pd.DataFrame({
        "datetime": ts,
        "open": op, "high": hi, "low": lo, "close": close,
        "vol": rng.integers(50, 1500, n_bars).astype(float),
        "spread": np.full(n_bars, 0.14),
    })
    return df


def ensure_data(data_dir: str = "data", synth_n: int = 60_000, seed: int = 42):
    """Return (df, source) where source is 'real' or 'synthetic'."""
    df = load_data(data_dir)
    if df is not None and len(df) > 1000:
        return df, "real"
    return make_synthetic(synth_n, seed=seed), "synthetic"


if __name__ == "__main__":
    df, src = ensure_data()
    print(f"source={src}  bars={len(df):,}  range={df.datetime.iloc[0]} .. {df.datetime.iloc[-1]}")
    print(df.head())
