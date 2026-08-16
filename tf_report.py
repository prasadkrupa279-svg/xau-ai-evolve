"""
tf_report.py — build a colour-coded HTML + CSV report of the strategy across
every timeframe (1m -> 1D) and every RR, on real 3-year XAUUSD data.
"""
import pandas as pd
from data_tools import load_data
from strategy_unified import get_precomp, run_backtest, UnifiedGenome

TFS = [("1m","1min",1),("3m","3min",3),("5m","5min",5),("15m","15min",15),
       ("30m","30min",30),("45m","45min",45),("1H","1h",60),("2H","2h",120),
       ("3H","3h",180),("4H","4h",240),("1D","1D",1440)]
RRS = [1.5, 2.0, 2.5, 3.0]


def resample_ohlc(df, rule):
    s = df.set_index("datetime")
    return pd.DataFrame({
        "open": s["open"].resample(rule).first(), "high": s["high"].resample(rule).max(),
        "low": s["low"].resample(rule).min(), "close": s["close"].resample(rule).last(),
        "vol": s["vol"].resample(rule).sum(), "spread": s["spread"].resample(rule).mean(),
    }).dropna(subset=["open","close"]).reset_index()


def run(tf_df, rr, trend):
    pre = get_precomp(tf_df)
    g = UnifiedGenome()
    g.asian_start, g.asian_end = 631, 728
    g.sweep_start, g.sweep_end, g.confirm_end = 736, 960, 1152
    g.range_atr, g.min_body, g.close_pos_min = 1.39, 0.59, 0.60
    g.min_stop_atr, g.stop_buffer = 1.32, 0.27
    g.use_daily = g.use_h1 = g.use_h4 = trend
    g.rr_target = rr
    return run_backtest(tf_df, g, pre)


def main():
    df1 = load_data("data")
    rows = []
    # 1m optimized champion (trend ON) for reference
    s = run(df1, 1.54, True)
    rows.append(("1m*", 1.54, s["trades"], s["winrate"], s["pf"], s["net_profit"], True))

    for name, rule, tfmin in TFS:
        df = resample_ohlc(df1, rule)
        if len(df) < 300:
            continue
        bas = max(1, round(207 / tfmin)); cd = max(0, round(8 / tfmin))
        for rr in RRS:
            pre = get_precomp(df)
            g = UnifiedGenome()
            g.asian_start, g.asian_end = 631, 728
            g.sweep_start, g.sweep_end, g.confirm_end = 736, 960, 1152
            g.bars_after_sweep, g.cooldown = bas, cd
            g.range_atr, g.min_body, g.close_pos_min = 1.39, 0.59, 0.60
            g.min_stop_atr, g.stop_buffer = 1.32, 0.27
            g.use_daily = g.use_h1 = g.use_h4 = False
            g.rr_target = rr
            st = run_backtest(df, g, pre)
            rows.append((name, rr, st["trades"], st["winrate"], st["pf"], st["net_profit"], False))

    # CSV
    pd.DataFrame(rows, columns=["TF","RR","trades","winrate","pf","net","champion"]).to_csv("tf_rr_report.csv", index=False)

    # best picks
    valid = [r for r in rows if r[2] >= 50]
    best_net = max(valid, key=lambda r: r[5]) if valid else None
    best_pf = max(valid, key=lambda r: r[4]) if valid else None

    # HTML
    def pf_cls(p):
        return "pf5" if p >= 1.5 else ("pf4" if p >= 1.2 else ("pf3" if p >= 1.0 else "pf2"))
    def net_cls(n):
        return "pos" if n > 0 else ("neg" if n < 0 else "")
    trs = []
    for tf, rr, t, wr, pf, net, champ in rows:
        champ_badge = '<span class="champ">★ optimized</span>' if champ else ""
        trs.append(
            f"<tr class='{'crow' if champ else ''}'>"
            f"<td>{tf} {champ_badge}</td><td>{rr}</td><td>{t}</td>"
            f"<td>{wr*100:.1f}%</td><td class='{pf_cls(pf)}'>{pf:.2f}</td>"
            f"<td class='{net_cls(net)}'>${net:,.0f}</td></tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XAUUSD — Timeframe × RR Results</title>
<style>
:root{{--bg:#0b1020;--c:#16203a;--t:#e6ecff;--m:#8ea0c8;--g:#22c55e;--r:#f87171}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(160deg,#0b1020,#121a30);color:var(--t);padding:18px}}
.wrap{{max-width:760px;margin:0 auto}}
h1{{font-size:1.2rem;margin-bottom:2px}}h1 span{{color:var(--g)}}
.sub{{color:var(--m);font-size:.82rem;margin-bottom:14px}}
.best{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}}
.best .b{{background:var(--c);border:1px solid #283450;border-radius:10px;padding:10px}}
.best .b .l{{font-size:.72rem;color:var(--m);text-transform:uppercase}}
.best .b .v{{font-size:.95rem;font-weight:700;margin-top:3px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem;background:var(--c);border-radius:10px;overflow:hidden}}
th,td{{padding:8px 9px;text-align:left;border-bottom:1px solid #243150}}
th{{color:var(--m);font-size:.72rem;text-transform:uppercase;background:#101a30}}
td.r,th.r{{text-align:right}}
.pf5{{color:#34d399;font-weight:700}}.pf4{{color:#86efac}}.pf3{{color:#fde68a}}.pf2{{color:var(--r)}}
.pos{{color:#34d399;font-weight:600}}.neg{{color:var(--r)}}
.crow{{background:rgba(34,197,94,.08)}}
.champ{{display:inline-block;background:rgba(56,189,248,.18);color:#7dd3fc;font-size:.62rem;padding:1px 6px;border-radius:8px;margin-left:4px}}
.note{{color:var(--m);font-size:.74rem;margin-top:14px;line-height:1.5}}
</style></head><body><div class="wrap">
<h1>🥇 XAUUSD — <span>Timeframe × RR</span> Results</h1>
<div class="sub">Real M1 data, 3 years (1,059,978 candles) · SL-guard · tight 1.32×ATR stops · trend filter OFF (fair cross-TF)</div>
<div class="best">
<div class="b"><div class="l">Highest net profit</div><div class="v">{best_net[0]} @ RR {best_net[1]} → PF {best_net[4]:.2f} · +${best_net[5]:,.0f} ({best_net[2]} trades)</div></div>
<div class="b"><div class="l">Highest profit factor (≥50 trades)</div><div class="v">{best_pf[0]} @ RR {best_pf[1]} → PF {best_pf[4]:.2f} · +${best_pf[5]:,.0f} ({best_pf[2]} trades)</div></div>
</div>
<table><thead><tr><th>Timeframe</th><th>RR</th><th class="r">Trades</th><th>Win Rate</th><th>Profit Factor</th><th class="r">Net</th></tr></thead>
<tbody>
{''.join(trs)}
</tbody></table>
<div class="note">
<b>★ 1m optimized</b> = trend filter ON (H1+H4), the GA champion (922 trades / 45.1% / PF 1.16).<br>
All other rows have trend filter OFF for a fair cross-timeframe comparison, so low TFs look weaker than they are with the filter.<br>
PF legend: <span class="pf5">≥1.5 strong</span> · <span class="pf4">1.2–1.5 good</span> · <span class="pf3">1.0–1.2 marginal</span> · <span class="pf2">&lt;1.0 losing</span>.<br>
4H/1D: too few/zero trades — this is an intraday (Asian→sweep→confirm) strategy by design.
</div>
</div></body></html>"""
    with open("tf_rr_report.html", "w") as f:
        f.write(html)
    print("wrote tf_rr_report.html and tf_rr_report.csv")
    print(f"best net: {best_net[:2]} PF {best_net[4]} +${best_net[5]} | best PF: {best_pf[:2]} PF {best_pf[4]}")


if __name__ == "__main__":
    main()
