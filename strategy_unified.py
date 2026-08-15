"""
strategy_unified.py
===================
Python port of the "Arena Unified Liquidity" Pine strategy (low-TF M1 model),
built so the AI agent can OPTIMISE its parameters on real data.

Mechanics (faithful):
  - Asian range build -> NY sweep -> confirmation candle (body% / close-pos / range>=ATR)
  - MTF trend bias (D / H1 / H4 EMA stacks, resampled from M1, shifted +1 HTF bar = no lookahead)
  - Entry at close; SL = min(sweepLow, low, close - ATR*minStopATR) - buffer
  - TP = entry +/- risk*RR; SL-GUARD first, gap-at-open handling, one trade at a time
  - Virtual commission per fill

Design choice: session windows (Asian/Sweep/Confirm) are EVOLVABLE, so the GA
discovers the alignment that works on THIS dataset's timezone -> no manual TZ fixing.
"""
from __future__ import annotations
import hashlib, json, random
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

COMMISSION_PCT = 0.02            # % per fill (Pine default)


@dataclass
class UnifiedGenome:
    asian_start: int = 1200; asian_end: int = 179      # 20:00 - 02:59 (wraps midnight)
    sweep_start: int = 180; sweep_end: int = 390        # 03:00 - 06:30
    confirm_end: int = 540                              # 09:00
    bars_after_sweep: int = 96
    range_atr: float = 1.30
    min_body: float = 0.35
    close_pos_min: float = 0.65
    min_stop_atr: float = 3.00
    rr_target: float = 3.00
    stop_buffer: float = 0.20
    cooldown: int = 2
    use_daily: bool = True; use_h1: bool = True; use_h4: bool = True

    def key(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]


def _in_window(mod: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        return (mod >= start) & (mod <= end)
    return (mod >= start) | (mod <= end)


_CACHE: dict = {}


class Precomp:
    def __init__(self, df: pd.DataFrame):
        self.n = len(df)
        self.o = df["open"].to_numpy(float); self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float); self.c = df["close"].to_numpy(float)
        self.spread = df["spread"].to_numpy(float)
        dt = pd.to_datetime(df["datetime"])
        self.mod = (dt.dt.hour * 60 + dt.dt.minute).to_numpy()
        self.times = dt.to_numpy()

        hl = self.h - self.l
        tr = np.maximum.reduce([hl, np.abs(self.h - np.append(self.c[0], self.c[:-1])),
                                np.abs(self.l - np.append(self.c[0], self.c[:-1]))])
        self.atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()

        s_h = pd.Series(self.h).shift(1); s_l = pd.Series(self.l).shift(1)
        self.refH3 = s_h.rolling(3, min_periods=1).max().to_numpy()
        self.refH10 = s_h.rolling(10, min_periods=1).max().to_numpy()
        self.refL3 = s_l.rolling(3, min_periods=1).min().to_numpy()
        self.refL10 = s_l.rolling(10, min_periods=1).min().to_numpy()

        self.daily_bull = self._htf("1D", lambda c, e: c > e[0])
        self.h1_bull = self._htf("1H", lambda c, e: (c > e[1]) & (e[0] > e[1]) & (e[1] > e[2]))
        self.h4_bull = self._htf("4H", lambda c, e: (c > e[1]) & (e[0] > e[1]) & (e[1] > e[2]))

    def _htf(self, rule, bullfn):
        cs = pd.Series(self.c, index=pd.Index(self.times)).resample(rule).last().dropna()
        e = [cs.ewm(span=p, adjust=False).mean() for p in (20, 50, 200)]
        bull = bullfn(cs, e).shift(1).fillna(False)
        hidx = bull.index.values.astype("datetime64[ns]")
        hvals = bull.to_numpy()
        pos = np.searchsorted(hidx, self.times, side="right") - 1
        out = np.zeros(self.n, dtype=bool)
        valid = pos >= 0
        out[valid] = hvals[np.clip(pos[valid], 0, len(hvals) - 1)]
        return out


def get_precomp(df: pd.DataFrame) -> Precomp:
    key = (len(df), str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1]))
    if key not in _CACHE:
        _CACHE[key] = Precomp(df)
    return _CACHE[key]


def _asian_carry(in_asian, h, l, n):
    """Pine-faithful: extend high/low during session, freeze outside."""
    ah = np.full(n, np.nan); al = np.full(n, np.nan)
    cur_h = cur_l = np.nan
    prev = False
    for i in range(n):
        a = bool(in_asian[i])
        if a and not prev:
            cur_h, cur_l = h[i], l[i]
        elif a:
            cur_h = max(cur_h, h[i]); cur_l = min(cur_l, l[i])
        ah[i] = cur_h; al[i] = cur_l
        prev = a
    return ah, al


def _carry_state(flag, ref_vals, n, life):
    idxs = np.where(flag, np.arange(n), -10**9)
    last = np.maximum.accumulate(idxs)
    age = np.arange(n) - last
    active = (last >= 0) & (age <= life)
    carry = np.where(last >= 0, ref_vals[np.clip(last, 0, n - 1)], np.nan)
    return active, carry


def run_backtest(df: pd.DataFrame, g: UnifiedGenome, pre: Precomp | None = None) -> dict:
    try:
        pre = pre or get_precomp(df)
        n = pre.n
        start = 250
        o, h, l, c, atr = pre.o, pre.h, pre.l, pre.c, pre.atr
        comm = COMMISSION_PCT / 100.0

        in_asian = _in_window(pre.mod, g.asian_start, g.asian_end)
        in_sweep = _in_window(pre.mod, g.sweep_start, g.sweep_end)
        in_confirm = _in_window(pre.mod, g.sweep_start, g.confirm_end)
        asia_high, asia_low = _asian_carry(in_asian, h, l, n)

        bull_sweep = in_sweep & ~np.isnan(asia_low) & (l < asia_low) & (c > asia_low)
        bear_sweep = in_sweep & ~np.isnan(asia_high) & (h > asia_high) & (c < asia_high)

        bull_active, bull_sweep_low = _carry_state(bull_sweep, l, n, g.bars_after_sweep)
        bear_active, bear_sweep_high = _carry_state(bear_sweep, h, n, g.bars_after_sweep)
        _, bull_break = _carry_state(bull_sweep, pre.refH3, n, g.bars_after_sweep)
        _, bear_break = _carry_state(bear_sweep, pre.refL3, n, g.bars_after_sweep)

        br = np.maximum(h - l, 1e-9)
        body_pct = np.abs(c - o) / br
        cpl = (c - l) / br; cps = (h - c) / br
        range_pass = (h - l) >= atr * g.range_atr
        strong_bull = (c > o) & (body_pct >= g.min_body) & (cpl >= g.close_pos_min)
        strong_bear = (c < o) & (body_pct >= g.min_body) & (cps >= g.close_pos_min)

        blong = np.ones(n, dtype=bool)
        bshort = np.ones(n, dtype=bool)
        if g.use_daily:
            blong &= pre.daily_bull; bshort &= ~pre.daily_bull
        if g.use_h1:
            blong &= pre.h1_bull; bshort &= ~pre.h1_bull
        if g.use_h4:
            blong &= pre.h4_bull; bshort &= ~pre.h4_bull

        long_sig = bull_active & in_confirm & range_pass & strong_bull & blong & (c > bull_break)
        short_sig = bear_active & in_confirm & range_pass & strong_bear & bshort & (c < bear_break)
        long_sig[:start] = False; short_sig[:start] = False

        rr, sm, buf = g.rr_target, g.min_stop_atr, g.stop_buffer
        wins = losses = trades = 0
        gw = gl = net = 0.0
        sig = np.where(long_sig | short_sig)[0]
        import os as _os
        if _os.environ.get("DBG"):
            import sys
            print(f"DBG g.asian={g.asian_start}-{g.asian_end} sweep={g.sweep_start}-{g.sweep_end} confirm_end={g.confirm_end}", file=sys.stderr)
            print(f"DBG in_asian={int(in_asian.sum())} in_sweep={int(in_sweep.sum())} asia_low_finite={int(np.isfinite(asia_low).sum())} bull_sweep={int(bull_sweep.sum())} bull_active={int(bull_active.sum())} strong_bull={int(strong_bull.sum())} [pre.daily={int(pre.daily_bull.sum())} h1={int(pre.h1_bull.sum())} h4={int(pre.h4_bull.sum())}] blong={int(blong.sum())} cgtbreak={int((c>bull_break).sum())} long={int(long_sig.sum())}", file=sys.stderr)
        last_exit = -10**9
        for e in sig:
            if e <= last_exit + g.cooldown:
                continue
            side = 1 if long_sig[e] else -1
            entry = c[e]
            if side > 0:
                ref_low = bull_sweep_low[e] if np.isfinite(bull_sweep_low[e]) else l[e]
                slp = min(ref_low, l[e], entry - atr[e] * sm) - buf
            else:
                ref_high = bear_sweep_high[e] if np.isfinite(bear_sweep_high[e]) else h[e]
                slp = max(ref_high, h[e], entry + atr[e] * sm) + buf
            risk = max(abs(entry - slp), 1e-6)
            tpp = entry + side * risk * rr
            j = e + 1; ep = None
            while j < n:
                oj, hj, lj = o[j], h[j], l[j]
                if side > 0:
                    if oj <= slp: ep = oj; break
                    if oj >= tpp: ep = oj; break
                    hsl, htp = lj <= slp, hj >= tpp
                    if hsl and htp:
                        ep = tpp if abs(hj - oj) < abs(oj - lj) else slp; break
                    elif hsl: ep = slp; break
                    elif htp: ep = tpp; break
                else:
                    if oj >= slp: ep = oj; break
                    if oj <= tpp: ep = oj; break
                    hsl, htp = hj >= slp, lj <= tpp
                    if hsl and htp:
                        ep = slp if abs(hj - oj) < abs(oj - lj) else tpp; break
                    elif hsl: ep = slp; break
                    elif htp: ep = tpp; break
                j += 1
            if ep is None:
                ep = c[n - 1]
            pnl = side * (ep - entry) - abs(entry) * comm - abs(ep) * comm
            trades += 1; net += pnl
            if pnl > 0: wins += 1; gw += pnl
            else: losses += 1; gl += abs(pnl)
            last_exit = j

        wr = wins / trades if trades else 0.0
        pf = gw / gl if gl > 0 else (10.0 if gw > 0 else 0.0)
        return {"trades": trades, "wins": wins, "losses": losses, "winrate": round(wr, 4),
                "pf": round(pf, 4), "net_profit": round(net, 2), "params": asdict(g), "key": g.key()}
    except Exception as e:
        print(f"[strategy_unified] backtest error: {e}")
        return {"trades": 0, "wins": 0, "losses": 0, "winrate": 0.0, "pf": 0.0,
                "net_profit": 0.0, "params": asdict(g), "key": g.key()}


def fitness(s: dict) -> float:
    """Reward MORE trades (user wants volume) + balanced WR + PF (capped to avoid
    tiny-sample overfit). Hard floor on trade count so it can't cherry-pick 35 trades."""
    tr, wr, pf, net = s.get("trades", 0), s.get("winrate", 0.0), s.get("pf", 0.0), s.get("net_profit", 0.0)
    if tr < 250:                                  # too few = statistically meaningless
        return 0.0
    f_wr = wr ** 1.5                              # higher WR rewarded, not dominantly
    f_pf = min(pf, 2.5) ** 1.1                    # cap PF -> no chasing tiny-sample 6.0
    if tr < 400:                                  # ramp trades factor 250->400
        f_tr = 0.4 + 0.6 * (tr - 250) / 150.0
    else:                                         # more trades = better, bonus to ~900
        f_tr = min(1.4, 1.0 + 0.4 * min(1.0, (tr - 400) / 500.0))
    f_net = 1.0 + 0.15 * np.tanh(net / 5000.0)
    return float(f_wr * f_pf * f_tr * f_net)


RANGES = {
    "asian_start": (0, 1440), "asian_end": (0, 1440), "sweep_start": (0, 1440),
    "sweep_end": (0, 1440), "confirm_end": (0, 1440), "bars_after_sweep": (8, 240),
    "range_atr": (0.3, 2.5), "min_body": (0.2, 0.7), "close_pos_min": (0.5, 0.8),
    "min_stop_atr": (1.0, 5.0), "rr_target": (1.5, 4.0), "stop_buffer": (0.0, 0.6), "cooldown": (0, 8),
}


def random_unified(rng=None):
    rng = rng or random
    d = asdict(UnifiedGenome())
    for k, (lo, hi) in RANGES.items():
        d[k] = rng.randint(int(lo), int(hi)) if isinstance(d[k], int) else round(rng.uniform(lo, hi), 2)
    d["use_daily"] = rng.random() < 0.7; d["use_h1"] = rng.random() < 0.7; d["use_h4"] = rng.random() < 0.7
    return UnifiedGenome(**d)


def mutate_unified(g, rate=0.3, rng=None):
    rng = rng or random
    d = asdict(g)
    for k, (lo, hi) in RANGES.items():
        if rng.random() < rate:
            step = (hi - lo) * 0.12
            nv = max(lo, min(hi, d[k] + rng.uniform(-step, step)))
            d[k] = int(round(nv)) if isinstance(d[k], int) else round(nv, 2)
    for b in ("use_daily", "use_h1", "use_h4"):
        if rng.random() < rate * 0.5:
            d[b] = not d[b]
    return UnifiedGenome(**d)


def crossover_unified(a, b, rng=None):
    rng = rng or random
    da, db = asdict(a), asdict(b)
    return UnifiedGenome(**{k: (va if rng.random() < 0.5 else db[k]) for k, va in da.items()})


if __name__ == "__main__":
    import time
    from data_tools import load_data
    df = load_data("data")
    print(f"data: {len(df):,} bars")
    t0 = time.time(); pre = get_precomp(df); print(f"precompute: {time.time()-t0:.1f}s")
    t0 = time.time(); s = run_backtest(df, UnifiedGenome(), pre)
    print(f"BASELINE (Pine defaults): trades={s['trades']} wr={s['winrate']} pf={s['pf']} net=${s['net_profit']} fit={fitness(s):.4f} ({time.time()-t0:.1f}s)")
