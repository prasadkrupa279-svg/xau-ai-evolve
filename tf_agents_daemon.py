"""
tf_agents_daemon.py — 11 AI agents, ONE PER TIMEFRAME (1m -> 1D).
Each agent evolves its OWN strategy params on its own resampled TF data,
RR pinned to that TF's original value, tracks its OWN trades/winrate/PF/net.
Infinite loop per agent. Results -> memory/tf_agents.json (read by dashboard).
"""
from __future__ import annotations
import os, time, json, random, threading, traceback
from dataclasses import asdict
import pandas as pd
from data_tools import load_data
from strategy_unified import (get_precomp, run_backtest, UnifiedGenome,
                              random_unified, crossover_unified, mutate_unified, RANGES)

TFS = [("1m", "1min", 3.0), ("5m", "5min", 3.0), ("15m", "15min", 2.5),
       ("1H", "1h", 2.0), ("4H", "4h", 2.0), ("1D", "1D", 1.0)]
MEM = "memory/tf_agents.json"


def resample(df, rule):
    s = df.set_index("datetime")
    return pd.DataFrame({
        "open": s["open"].resample(rule).first(), "high": s["high"].resample(rule).max(),
        "low": s["low"].resample(rule).min(), "close": s["close"].resample(rule).last(),
        "vol": s["vol"].resample(rule).sum(), "spread": s["spread"].resample(rule).mean(),
    }).dropna(subset=["open", "close"]).reset_index()


def score(s):
    """rewards winrate × PF × trades (no hard floor -> low-trade TFs still show)."""
    pf = max(s.get("pf", 0), 0); wr = s.get("winrate", 0); tr = s.get("trades", 0)
    import math
    return pf * wr * (1 + math.log1p(tr) / 4.0)


class Hub:
    def __init__(self):
        self.results = {}      # tf -> stats
        self.lock = threading.Lock()

    def update(self, tf, name, rr, s, evals, dups, params):
        sc = score(s)
        s = dict(s); s["score"] = round(sc, 4)
        with self.lock:
            cur = self.results.get(tf)
            if cur is None or sc >= cur.get("score", -1):
                self.results[tf] = {"name": name, "tf": tf, "rr": rr,
                                    "trades": s["trades"], "winrate": s["winrate"],
                                    "pf": s["pf"], "net_profit": s["net_profit"],
                                    "score": s["score"], "evals": evals, "dups": dups,
                                    "params": params, "updated": time.strftime("%H:%M:%S")}
            else:
                # keep counts fresh even if not a new best
                self.results[tf]["evals"] = evals
                self.results[tf]["dups"] = dups


def agent(name, tf, rule, rr, df_m1, hub, seed):
    rng = random.Random(seed)
    df = resample(df_m1, rule)
    if len(df) < 300:
        print(f"[{name}] too few bars ({len(df)})", flush=True); return
    pre = get_precomp(df)
    del df                      # free resampled frame (precomp holds its own arrays)
    tfmin = int(round(1440 / max(1, df_m1["datetime"].dt.minute.diff().mode()[0]))) if False else 1
    seen = set(); elites = []; best_sc = -1; evals = 0; dups = 0
    print(f"[{name}] online | bars={pre.n:,} | RR pinned 1:{rr}", flush=True)
    while True:
        try:
            if len(elites) >= 2:
                g = mutate_unified(crossover_unified(rng.choice(elites), rng.choice(elites), rng), 0.3, rng)
            else:
                g = mutate_unified(random_unified(rng), 0.4, rng)
            # scale session/timing to TF bar size
            tf_minutes = {"1min":1,"3min":3,"5min":5,"15min":15,"30min":30,"45min":45,"1h":60,"2h":120,"3h":180,"4h":240,"1D":1440}[rule]
            g.bars_after_sweep = max(1, round(g.bars_after_sweep * 1 / tf_minutes)) if False else g.bars_after_sweep
            g.rr_target = rr
            k = g.key()
            if k in seen:
                dups += 1; continue
            seen.add(k); evals += 1
            s = run_backtest(None, g, pre)   # pre passed -> df not needed (freed)
            hub.update(tf, name, rr, s, evals, dups, asdict(g))
            sc = score(s)
            if sc > best_sc:
                best_sc = sc
            elites.append(g)
            if len(elites) > 20:
                elites = elites[-20:]
            if evals % 5 == 0:
                with hub.lock:
                    r = hub.results.get(tf, {})
                print(f"[{name}] evals={evals} dups={dups} | best trades={r.get('trades')} "
                      f"wr={r.get('winrate')} pf={r.get('pf')} net={r.get('net_profit')}", flush=True)
        except Exception as e:
            print(f"[{name}] err: {e}", flush=True); time.sleep(2)


def main():
    df = load_data("data")
    print(f"[main] M1 data={len(df):,} bars -> launching {len(TFS)} TF agents (1m..1D)", flush=True)
    hub = Hub()
    for i, (tf, rule, rr) in enumerate(TFS):
        threading.Thread(target=agent, args=(f"Agent-{tf}", tf, rule, rr, df, hub, 200 + i), daemon=True).start()
    while True:
        time.sleep(10)
        with hub.lock:
            data = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "agents": dict(hub.results)}
        try:
            from util import safe_write_json
            safe_write_json(MEM, data)
        except Exception:
            pass
        n = len(data["agents"])
        print(f"[main] agents_reporting={n}/{len(TFS)} | saved {MEM}", flush=True)


if __name__ == "__main__":
    main()
