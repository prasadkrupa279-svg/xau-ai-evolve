"""
full_data_trades.py — FULL 1.06M-bar (3-yr) backtest per TF, pushing MAX trades.
Runs locally (sandbox has RAM). Loose filters -> high trade volume + best WR/PF found.
"""
import time, random
import pandas as pd
from data_tools import load_data
from strategy_unified import get_precomp, run_backtest, UnifiedGenome, mutate_unified

TFS = [("1m","1min",3.0),("3m","3min",3.0),("5m","5min",3.0),("15m","15min",2.5),
       ("30m","30min",2.0),("45m","45min",2.0),("1H","1h",2.0),("2H","2h",2.0),
       ("3H","3h",2.0),("4H","4h",2.0),("1D","1D",1.0)]

def resample(df, rule):
    s = df.set_index("datetime")
    return pd.DataFrame({
        "open":s["open"].resample(rule).first(),"high":s["high"].resample(rule).max(),
        "low":s["low"].resample(rule).min(),"close":s["close"].resample(rule).last(),
        "vol":s["vol"].resample(rule).sum(),"spread":s["spread"].resample(rule).mean(),
    }).dropna(subset=["open","close"]).reset_index()

def loose(rr, tfmin):
    g = UnifiedGenome()
    g.ind_conf=1; g.range_atr=0.3; g.min_body=0.2; g.close_pos_min=0.5
    g.use_daily=False; g.use_h1=False; g.use_h4=False
    g.bars_after_sweep=max(40, 200//max(1,tfmin//60) if tfmin>=60 else 200)
    g.cooldown=0; g.rr_target=rr
    return g

df = load_data("data")  # FULL 1.06M
print(f"FULL data: {len(df):,} bars ({df['datetime'].iloc[0].date()} -> {df['datetime'].iloc[-1].date()})")
print(f"{'TF':<6}{'RR':<5}{'trades':>8}{'WR':>8}{'PF':>8}{'net':>10}")
print("-"*45)
rng = random.Random(3)
TOT=0
for name,rule,rr in TFS:
    tdf = resample(df, rule)
    if len(tdf) < 30:
        print(f"{name:<6} (too few: {len(tdf)})"); continue
    pre = get_precomp(tdf)
    tfmin = {"1min":1,"3min":3,"5min":5,"15min":15,"30min":30,"45min":45,"1h":60,"2h":120,"3h":180,"4h":240,"1D":1440}[rule]
    # search loose configs (max trades with WR>25%)
    best=None
    for i in range(12):
        g = loose(rr, tfmin) if i==0 else mutate_unified(loose(rr,tfmin), 0.4, rng)
        g.rr_target=rr
        s = run_backtest(tdf, g, pre)
        s["fit"] = s["trades"]*(0.3+0.7*s["winrate"])*(min(s["pf"],3))  # trades-dominant
        if best is None or s["fit"]>best["fit"]:
            best=s
    TOT+=best["trades"]
    print(f"{name:<6}{rr:<5}{best['trades']:>8}{best['winrate']*100:>7.0f}%{best['pf']:>8.2f}{best['net_profit']:>10.0f}")
    del pre
print("-"*45)
print(f"TOTAL trades across 11 TFs: {TOT}")
