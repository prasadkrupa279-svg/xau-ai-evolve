"""
tf_fixed_rr.py — run each timeframe at the RR the Pine script pre-assigns to it.
RR map (from Pine): low-TF(1m/3m/5m)=3.0, 15m=2.5, 30m/45m/1H/2H/3H/4H=2.0, 1D=1.0
"""
import pandas as pd
from data_tools import load_data
from strategy_unified import get_precomp, run_backtest, UnifiedGenome

TFS = [("1m","1min",1,3.0),("3m","3min",3,3.0),("5m","5min",5,3.0),("15m","15min",15,2.5),
       ("30m","30min",30,2.0),("45m","45min",45,2.0),("1H","1h",60,2.0),("2H","2h",120,2.0),
       ("3H","3h",180,2.0),("4H","4h",240,2.0),("1D","1D",1440,1.0)]

def resample_ohlc(df, rule):
    s = df.set_index("datetime")
    return pd.DataFrame({
        "open":s["open"].resample(rule).first(),"high":s["high"].resample(rule).max(),
        "low":s["low"].resample(rule).min(),"close":s["close"].resample(rule).last(),
        "vol":s["vol"].resample(rule).sum(),"spread":s["spread"].resample(rule).mean(),
    }).dropna(subset=["open","close"]).reset_index()

df1 = load_data("data")
print(f"{'TF':<6}| {'RR':<4}| {'trades':>7}| {'WR':>7}| {'PF':>6}| {'net':>10}")
print("-"*48)
out=[]
for name,rule,tfmin,rr in TFS:
    df=resample_ohlc(df1,rule)
    if len(df)<300:
        print(f"{name:<6}| too few bars"); continue
    pre=get_precomp(df)
    g=UnifiedGenome()
    g.asian_start,g.asian_end=631,728
    g.sweep_start,g.sweep_end,g.confirm_end=736,960,1152
    g.bars_after_sweep=max(1,round(207/tfmin)); g.cooldown=max(0,round(8/tfmin))
    g.range_atr,g.min_body,g.close_pos_min=1.39,0.59,0.60
    g.min_stop_atr,g.stop_buffer=1.32,0.27
    g.use_daily=g.use_h1=g.use_h4=False
    g.rr_target=rr
    s=run_backtest(df,g,pre)
    print(f"{name:<6}| {rr:<4}| {s['trades']:>7}| {s['winrate']*100:>6.1f}%| {s['pf']:>6.2f}| ${s['net_profit']:>9.0f}")
    out.append((name,rr,s['trades'],s['winrate'],s['pf'],s['net_profit']))
pd.DataFrame(out,columns=["TF","RR","trades","winrate","pf","net"]).to_csv("tf_fixed_rr.csv",index=False)
