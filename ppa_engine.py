"""
ppa_engine.py
=============
AB-Touch POI backtest engine.  Faithful, no-look-ahead, SL-guard first.

Pipeline (all decisions use CLOSED bars; fills happen at the NEXT bar's OPEN):
  1. net vote per bar = sum of the agent's enabled 37-bit mask votes.
  2. zone = active (<480 bars) swing-pivot level OR Fair-Value-Gap that price
     has wicked back into (touch + wick)  -> the "AB-Touch POI".
  3. trend filter (ptr): off / EMA50>EMA100 / EMA100>EMA200.
  4. fire when |net| >= ind_conf and zone + trend agree on direction.
  5. fill at next open (buy=open+spread, sell=open-spread).
  6. resolve exit by scanning forward: SL-GUARD FIRST (same-bar SL beats TP),
     else TP = full win. One position at a time. No trailing, no partials.

Contract model: sl/tp are PRICE distances ($). 1 lot XAUUSD = 100 oz, so
a $0.5 SL = $50 risk, a $2.5 TP = $250 reward per 1.0 lot.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from genome import Genome

CONTRACT = 100.0      # oz per 1.0 lot
SPREAD_DEFAULT = 0.14
MAX_HOLD = 2000
WARMUP = 250
ZONE_LIFE = 480


def _ema(s: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(s).ewm(span=p, adjust=False).mean().to_numpy()


def _swing_levels(low: np.ndarray, high: np.ndarray, psw: int):
    """Return carry-forward last swing-low / swing-high price + their age (bars)."""
    n = len(low)
    df = pd.DataFrame({"low": low, "high": high})
    rollmin = df["low"].rolling(psw, min_periods=1).min()
    rollmax = df["high"].rolling(psw, min_periods=1).max()
    is_swing_low = (df["low"] == rollmin).to_numpy()
    is_swing_high = (df["high"] == rollmax).to_numpy()

    sl_series = pd.Series(np.where(is_swing_low, low, np.nan))
    sh_series = pd.Series(np.where(is_swing_high, high, np.nan))
    sl_carry = sl_series.ffill().to_numpy()
    sh_carry = sh_series.ffill().to_numpy()

    # age since last pivot: bars since last True
    def _age(flag):
        idx = np.where(flag, np.arange(n), np.nan)
        carry = pd.Series(idx).ffill().to_numpy()
        age = np.arange(n) - np.nan_to_num(carry, nan=0.0)
        return age
    return sl_carry, sh_carry, _age(is_swing_low), _age(is_swing_high)


def _fvg_zones(low: np.ndarray, high: np.ndarray):
    """Bullish/bearish FVG carry zones (3-bar gap)."""
    n = len(low)
    bull_fvg = low[2:] > high[:-2]            # gap up between bar i-2 high and bar i low
    bear_fvg = high[2:] < low[:-2]
    bull = np.zeros(n, dtype=bool); bull[2:] = bull_fvg
    bear = np.zeros(n, dtype=bool); bear[2:] = bear_fvg
    # zone midpoints carried forward
    bull_zone = pd.Series(np.where(bull, (high[:-2].tolist()[:1] + list(high[:-2]))[:0] if False else np.nan, np.nan)).to_numpy()
    # simpler: carry the gap midpoint
    bull_mid = np.where(bull, (low[2:] if False else np.nan), np.nan)
    # build arrays aligned
    bm = np.full(n, np.nan)
    tm = np.full(n, np.nan)
    for i in range(2, n):
        if low[i] > high[i - 2]:
            bm[i] = 0.5 * (high[i - 2] + low[i])
        if high[i] < low[i - 2]:
            tm[i] = 0.5 * (low[i - 2] + high[i])
    bm_carry = pd.Series(bm).ffill().to_numpy()
    tm_carry = pd.Series(tm).ffill().to_numpy()
    return bm_carry, tm_carry


def run_backtest(df: pd.DataFrame, votes: np.ndarray, g: Genome,
                 lot: float = 1.0, spread: float | None = None,
                 max_hold: int = MAX_HOLD) -> dict:
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    n = len(df)
    if n < WARMUP + max_hold + 10:
        return _empty_stats()
    sp = spread if spread is not None else float(df["spread"].median() or SPREAD_DEFAULT)

    # ---- net votes per bar (closed bar), used at NEXT open
    mask = g.mask_array()
    net = (votes.astype(np.int16) * mask.astype(np.int16)).sum(axis=1)

    # ---- trend filter
    e50, e100, e200 = _ema(c, 50), _ema(c, 100), _ema(c, 200)
    if g.ptr == 100:
        trend_up = e50 > e100; trend_dn = e50 < e100
    elif g.ptr == 200:
        trend_up = e100 > e200; trend_dn = e100 < e200
    else:
        trend_up = np.ones(n, bool); trend_dn = np.ones(n, bool)

    # ---- zones (swing pivots + optional FVG)
    sl_carry, sh_carry, sl_age, sh_age = _swing_levels(l, h, max(3, g.psw))
    zlong_age_ok = sl_age <= ZONE_LIFE
    zshort_age_ok = sh_age <= ZONE_LIFE
    long_touch = zlong_age_ok & (l <= sl_carry + g.pwk) & (c >= sl_carry - g.pwk)
    short_touch = zshort_age_ok & (h >= sh_carry - g.pwk) & (c <= sh_carry + g.pwk)
    if g.use_ppa:                                   # OR-in FVG touch zones
        bm_carry, tm_carry = _fvg_zones(l, h)
        long_touch |= ~np.isnan(bm_carry) & (np.abs(l - bm_carry) <= g.pwk)
        short_touch |= ~np.isnan(tm_carry) & (np.abs(h - tm_carry) <= g.pwk)
    long_touch = np.nan_to_num(long_touch, nan=False).astype(bool)
    short_touch = np.nan_to_num(short_touch, nan=False).astype(bool)

    # ---- entry signals (decision at bar k, fill open[k+1])
    ic = g.ind_conf
    long_sig = long_touch & trend_up & (net >= ic)
    short_sig = short_touch & trend_dn & (net <= -ic)
    long_sig[:WARMUP] = False; short_sig[:WARMUP] = False

    # ---- sequential trade resolution, SL-guard first
    sl_px = g.sl; tp_px = g.tp
    wins = losses = timeouts = 0
    gross_win = gross_loss = 0.0
    net_pnl = 0.0
    k = WARMUP
    while k < n - 1:
        side = 0
        if long_sig[k]:
            side = 1
        elif short_sig[k]:
            side = -1
        if side == 0:
            k += 1; continue

        eb = k + 1                              # entry bar
        entry = o[eb] + (sp if side > 0 else -sp)
        if side > 0:
            tp_price = entry + tp_px; sl_price = entry - sl_px
        else:
            tp_price = entry - tp_px; sl_price = entry + sl_px

        end = min(n, eb + 1 + max_hold)
        hh = h[eb + 1:end]; ll = l[eb + 1:end]
        if len(hh) == 0:
            k += 1; continue
        if side > 0:
            sl_hit = ll <= sl_price; tp_hit = hh >= tp_price
        else:
            sl_hit = hh >= sl_price; tp_hit = ll <= tp_price
        any_hit = sl_hit | tp_hit

        if not any_hit.any():                   # timeout -> mark-to-close
            exits_close = c[end - 1]
            pnl = (exits_close - entry) * side * CONTRACT * lot
            timeouts += 1; net_pnl += pnl
            k = end; continue

        j = int(np.argmax(any_hit))             # first hit bar
        rel = eb + 1 + j
        if sl_hit[j]:                           # SL-GUARD: SL wins even if TP also hit
            losses += 1
            gross_loss += sl_px * CONTRACT * lot
            net_pnl -= sl_px * CONTRACT * lot
        else:
            wins += 1
            gross_win += tp_px * CONTRACT * lot
            net_pnl += tp_px * CONTRACT * lot
        k = rel + 1

    trades = wins + losses + timeouts
    winrate = wins / trades if trades else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
    return {
        "trades": trades, "wins": wins, "losses": losses, "timeouts": timeouts,
        "winrate": round(winrate, 4), "pf": round(pf, 4),
        "net_profit": round(net_pnl, 2), "sl": g.sl, "tp": g.tp, "rr": round(g.tp / g.sl, 2),
    }


def _empty_stats() -> dict:
    return {"trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
            "winrate": 0.0, "pf": 0.0, "net_profit": 0.0, "sl": 0, "tp": 0, "rr": 0}


if __name__ == "__main__":
    import time, random
    from data_tools import ensure_data
    from indicators_lib import compute_votes
    from genome import random_genome, fitness
    df, src = ensure_data()
    v, _ = compute_votes(df)
    print(f"data={src} bars={len(df):,}")
    rng = random.Random(7)
    for sl in (0.5, 1.0):
        g = random_genome(sl, rng)
        t0 = time.time()
        s = run_backtest(df, v, g)
        print(f"  sl={sl} key={g.key()} ind_conf={g.ind_conf} enabled={bin(g.enabled).count('1')} "
              f"-> trades={s['trades']} wr={s['winrate']} pf={s['pf']} net=${s['net_profit']} "
              f"fit={round(fitness(s),4)} ({round(time.time()-t0,2)}s)")
