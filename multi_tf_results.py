"""
multi_tf_results.py
===================
Run the Arena Unified strategy on EVERY timeframe (1m -> 1D) at multiple RR values.
M1 data is resampled to each TF. Trend filter OFF for a fair cross-TF comparison.
Session windows = AI-optimized (broker-server time). SL-guard, one-trade-lock intact.
"""
import pandas as pd
from data_tools import load_data
from strategy_unified import get_precomp, run_backtest, UnifiedGenome


def resample_ohlc(df, rule):
    s = df.set_index("datetime")
    out = pd.DataFrame({
        "open":   s["open"].resample(rule).first(),
        "high":   s["high"].resample(rule).max(),
        "low":    s["low"].resample(rule).min(),
        "close":  s["close"].resample(rule).last(),
        "vol":    s["vol"].resample(rule).sum(),
        "spread": s["spread"].resample(rule).mean(),
    }).dropna(subset=["open", "close"]).reset_index()
    return out


TFS = [("1m", "1min", 1), ("3m", "3min", 3), ("5m", "5min", 5), ("15m", "15min", 15),
       ("30m", "30min", 30), ("45min", "45min", 45), ("1H", "1h", 60), ("2H", "2h", 120),
       ("3H", "3h", 180), ("4H", "4h", 240), ("1D", "1D", 1440)]
RRS = [1.5, 2.0, 2.5, 3.0]

df1 = load_data("data")
print(f"M1 base data: {len(df1):,} bars (3 years)\n")
hdr = f"{'TF':<6}| {'RR':<4}| {'trades':>7}| {'WR':>7}| {'PF':>6}| {'net':>10}| {'avg stop':>9}"
print(hdr); print("-" * len(hdr))

for name, rule, tfmin in TFS:
    df = resample_ohlc(df1, rule)
    if len(df) < 300:
        print(f"{name:<6}| too few bars ({len(df)})"); continue
    pre = get_precomp(df)
    bas = max(1, round(207 / tfmin)); cd = max(0, round(8 / tfmin))
    for rr in RRS:
        g = UnifiedGenome()
        g.asian_start, g.asian_end = 631, 728
        g.sweep_start, g.sweep_end, g.confirm_end = 736, 960, 1152
        g.bars_after_sweep, g.range_atr = bas, 1.39
        g.min_body, g.close_pos_min = 0.59, 0.60
        g.min_stop_atr, g.stop_buffer, g.cooldown = 1.32, 0.27, cd
        g.use_daily = g.use_h1 = g.use_h4 = False   # trend OFF -> fair cross-TF compare
        g.rr_target = rr
        s = run_backtest(df, g, pre)
        avg_stop = 1.32  # ATR multiple (constant preset)
        print(f"{name:<6}| {rr:<4}| {s['trades']:>7}| {s['winrate']*100:>6.1f}%| {s['pf']:>6.2f}| ${s['net_profit']:>9.0f}| {avg_stop:>6.2f}ATR")
    print()
