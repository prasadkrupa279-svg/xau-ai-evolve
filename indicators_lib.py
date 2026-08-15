"""
indicators_lib.py
=================
37 vote-sources built from 15 indicators (multiple periods). Each source casts one
vote per bar: +1 (bull), -1 (bear), 0 (neutral).

ALL votes are computed on CLOSED bars and the caller uses the *previous* bar's vote
to decide at the current bar (see ppa_engine) -> strict no-look-ahead.

Public API:
    votes, names = compute_votes(df)     # votes: int8 ndarray (n_bars, 37); names: list[str]
"""
from __future__ import annotations
import numpy as np
import pandas as pd

VOTE_DIM = 37

# ------------------------------------------------------------------ helpers
def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _rsi(s: pd.Series, p: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)

def _rollmax(s: pd.Series, p: int): return s.rolling(p, min_periods=1).max()
def _rollmin(s: pd.Series, p: int): return s.rolling(p, min_periods=1).min()
def _rollsma(s: pd.Series, p: int): return s.rolling(p, min_periods=1).mean()

def _sign(x, neutral=0.0):
    """+1 / -1 / 0 with a small neutral band."""
    out = np.where(x > neutral, 1, np.where(x < -neutral, -1, 0))
    return out

def _cmp(a, b):
    """+1 if a>b, -1 if a<b, 0 eq."""
    return np.where(a > b, 1, np.where(a < b, -1, 0))


# ------------------------------------------------------------------ indicator value builders
def _build_values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["vol"].astype(float)
    n = len(df)
    vals: dict[str, np.ndarray] = {}
    nan = np.full(n, np.nan)

    # --- RSI (5,9,14,21,30) -> vote: >50 bull
    for p in (5, 9, 14, 21, 30):
        vals[f"rsi{p}"] = _rsi(c, p).to_numpy()

    # --- Stochastic %K (9,14,21) -> vote: K>D
    for p in (9, 14, 21):
        ll = _rollmin(l, p); hh = _rollmax(h, p)
        k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        d = _rollsma(k, 3)
        vals[f"stoch{p}"] = (k.to_numpy() - d.to_numpy())

    # --- Williams %R (9,14,21) -> vote: >-50 bull (closer to 0)
    for p in (9, 14, 21):
        ll = _rollmin(l, p); hh = _rollmax(h, p)
        vals[f"wpr{p}"] = (-100 * (hh - c) / (hh - ll).replace(0, np.nan)).to_numpy()

    # --- MFI (14,21,30) -> >50 bull
    for p in (14, 21, 30):
        tp = (h + l + c) / 3.0
        rmf = tp * v
        pos = tp.diff().clip(lower=0).fillna(0)
        neg = (-tp.diff()).clip(upper=0).fillna(0)
        mfp = (rmf * pos).rolling(p, min_periods=1).sum()
        mfn = (rmf * neg).rolling(p, min_periods=1).sum()
        mfi = 100 - 100 / (1 + mfp / mfn.replace(0, np.nan))
        vals[f"mfi{p}"] = (mfi.fillna(50.0) - 50.0).to_numpy()

    # --- CMF Chaikin (20,30) -> >0 bull
    for p in (20, 30):
        mfmult = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        mfv = mfmult * v
        cmf = mfv.rolling(p, min_periods=1).sum() / v.rolling(p, min_periods=1).sum().replace(0, np.nan)
        vals[f"cmf{p}"] = cmf.fillna(0.0).to_numpy()

    # --- Force Index (13,21,34) -> >0 bull
    for p in (13, 21, 34):
        fi = (c.diff() * v).ewm(span=p, adjust=False).mean()
        vals[f"force{p}"] = fi.fillna(0.0).to_numpy()

    # --- OBV slope (10,20) -> rising bull
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
    for p in (10, 20):
        vals[f"obv{p}"] = (pd.Series(obv) - pd.Series(obv).shift(p)).fillna(0.0).to_numpy()

    # --- VWAP deviation (20,30) -> price>vwap bull
    for p in (20, 30):
        tp = (h + l + c) / 3.0
        vol = v.replace(0, np.nan)
        vwap = (tp * vol).rolling(p, min_periods=1).sum() / vol.rolling(p, min_periods=1).sum()
        vals[f"vwap{p}"] = ((c - vwap) / c * 100).fillna(0.0).to_numpy()

    # --- Aroon (14,25) -> Up>Down bull  (VECTORISED: no per-window Python call)
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    _carr = c.to_numpy(dtype=float)
    _n = len(_carr)
    for p in (14, 25):
        if _n > p:
            _sw = _swv(_carr, p + 1)             # (n-p, p+1) windows, oldest..newest
            up_idx = _sw.argmax(axis=1)          # position of high (0=oldest .. p=newest)
            dn_idx = _sw.argmin(axis=1)
            up = np.zeros(_n); dn = np.zeros(_n)
            up[p:] = up_idx / p * 100.0
            dn[p:] = dn_idx / p * 100.0
        else:
            up = np.zeros(_n); dn = np.zeros(_n)
        vals[f"aroon{p}"] = (up - dn)

    # --- TSI (13,21,30) -> >0 bull  (r,s smoothing)
    m = c.diff()
    for p in (13, 21, 30):
        m1 = m.ewm(span=p, adjust=False).mean().ewm(span=p * 2, adjust=False).mean()
        absm = m.abs().ewm(span=p, adjust=False).mean().ewm(span=p * 2, adjust=False).mean()
        tsi = 100 * m1 / absm.replace(0, np.nan)
        vals[f"tsi{p}"] = tsi.fillna(0.0).to_numpy()

    # --- Ultimate Oscillator (28) -> >50 bull
    p = 28
    tr7 = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    bp = c - np.minimum(l, c.shift())
    uo = 100 * (4 * bp.rolling(7, min_periods=1).mean() / tr7.rolling(7, min_periods=1).mean().replace(0, np.nan)
                + 2 * bp.rolling(14, min_periods=1).mean() / tr7.rolling(14, min_periods=1).mean().replace(0, np.nan)
                + bp.rolling(p, min_periods=1).mean() / tr7.rolling(p, min_periods=1).mean().replace(0, np.nan)) / 7
    vals["ult28"] = (uo.fillna(50.0) - 50.0).to_numpy()

    # --- Balance of Power -> >0 bull
    rng_ = (h - l).replace(0, np.nan)
    vals["bop"] = ((c - o) / rng_).fillna(0.0).to_numpy()

    # --- Vortex (14,21) -> VI+>VI- bull
    for p in (14, 21):
        vm = (h - l.shift()).abs()
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        vip = vm.rolling(p, min_periods=1).sum() / tr.rolling(p, min_periods=1).sum().replace(0, np.nan)
        vmn = (l - h.shift()).abs().rolling(p, min_periods=1).sum() / tr.rolling(p, min_periods=1).sum().replace(0, np.nan)
        vals[f"vort{p}"] = (vip - vmn).fillna(0.0).to_numpy()

    # --- ZLEMA (21,30,50) -> price>zlema bull
    for p in (21, 30, 50):
        lag = c.shift(int(p / 2)).fillna(c.iloc[0])
        zlema = (2 * c - lag).ewm(span=p, adjust=False).mean()
        vals[f"zlema{p}"] = ((c - zlema) / c * 100).fillna(0.0).to_numpy()

    # --- Elder Ray (13,21) -> sign(bull_power + bear_power)
    for p in (13, 21):
        e = _ema(c, p)
        bull = h - e
        bear = l - e
        vals[f"elder{p}"] = (bull + bear).fillna(0.0).to_numpy()

    return vals


# ordered list of the 37 sources (matches GENOME mask order)
VOTE_NAMES: list[str] = [
    "rsi5", "rsi9", "rsi14", "rsi21", "rsi30",
    "stoch9", "stoch14", "stoch21",
    "wpr9", "wpr14", "wpr21",
    "mfi14", "mfi21", "mfi30",
    "cmf20", "cmf30",
    "force13", "force21", "force34",
    "obv10", "obv20",
    "vwap20", "vwap30",
    "aroon14", "aroon25",
    "tsi13", "tsi21", "tsi30",
    "ult28",
    "bop",
    "vort14", "vort21",
    "zlema21", "zlema30", "zlema50",
    "elder13", "elder21",
]
assert len(VOTE_NAMES) == VOTE_DIM


def _vote_for(name: str, raw: np.ndarray) -> np.ndarray:
    """Map a raw indicator array to {-1,0,1} votes per the source's convention."""
    raw = np.asarray(raw, dtype=float)
    with np.errstate(invalid="ignore"):
        if name.startswith("rsi"):
            return _sign(raw - 50.0)
        if name.startswith("stoch") or name.startswith("aroon"):
            return _sign(raw)
        if name.startswith("wpr"):
            return _sign(raw + 50.0)  # >-50 bull
        if name.startswith("mfi"):
            return _sign(raw)           # already centred at 0
        if name in ("cmf20", "cmf30", "force13", "force21", "force34",
                    "obv10", "obv20", "tsi13", "tsi21", "tsi30",
                    "bop", "vort14", "vort21", "ult28",
                    "vwap20", "vwap30", "zlema21", "zlema30", "zlema50"):
            return _sign(raw)
        if name.startswith("elder"):
            return _sign(raw)
    return np.zeros(len(raw), dtype=np.int8)


def compute_votes(df: pd.DataFrame):
    """Return (votes int8 (n,37), names). NaN-safe; early bars -> 0."""
    vals = _build_values(df)
    n = len(df)
    votes = np.zeros((n, VOTE_DIM), dtype=np.int8)
    for j, name in enumerate(VOTE_NAMES):
        v = _vote_for(name, vals[name])
        v = np.nan_to_num(v, nan=0.0).astype(np.int8)
        votes[:, j] = v
    return votes, list(VOTE_NAMES)


if __name__ == "__main__":
    from data_tools import ensure_data
    df, _ = ensure_data()
    v, names = compute_votes(df)
    nz = (v != 0).mean(axis=0)
    print("votes shape:", v.shape, "| nonzero ratio/col mean:", round(float(nz.mean()), 3))
    print("bull bias/col:", (v == 1).sum(0)[:5], "... bear:", (v == -1).sum(0)[:5])
