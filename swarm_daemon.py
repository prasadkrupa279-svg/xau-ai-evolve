"""
swarm_daemon.py — MAX agents (100s) WITHOUT crash.

Secret: ONE shared precomp per TF (memory-bounded) + many lightweight agent threads
that all evaluate against that shared precomp, with per-TF GLOBAL dedup
(no two agents repeat a genome). -> hundreds of agents, no OOM.

  - 6 TFs (memory-safe set): 1m, 5m, 15m, 1H, 4H, 1D
  - AGENTS_PER_TF agents per TF  (default 50  -> 300 agents total)
  - RR pinned per TF (original). Fitness rewards winrate × PF × trades.
  - Infinite loop. Results -> memory/tf_agents.json (dashboard reads it).
"""
from __future__ import annotations
import os, time, json, math, random, threading
from dataclasses import asdict
import pandas as pd
from data_tools import load_data
from strategy_unified import (get_precomp, run_backtest, UnifiedGenome,
                              random_unified, crossover_unified, mutate_unified)

TFS = [("1m", "1min", 3.0), ("5m", "5min", 3.0), ("15m", "15min", 2.5),
       ("1H", "1h", 2.0), ("4H", "4h", 2.0), ("1D", "1D", 1.0)]
AGENTS_PER_TF = int(os.environ.get("AGENTS_PER_TF", "50"))   # 6*50 = 300 agents
MAX_BARS = int(os.environ.get("MAX_BARS", "0"))              # cap bars/TF to save RAM (0=all)
MEM = "memory/tf_agents.json"


def resample(df, rule):
    s = df.set_index("datetime")
    return pd.DataFrame({
        "open": s["open"].resample(rule).first(), "high": s["high"].resample(rule).max(),
        "low": s["low"].resample(rule).min(), "close": s["close"].resample(rule).last(),
        "vol": s["vol"].resample(rule).sum(), "spread": s["spread"].resample(rule).mean(),
    }).dropna(subset=["open", "close"]).reset_index()


def score(s):
    pf = max(s.get("pf", 0), 0); wr = s.get("winrate", 0); tr = s.get("trades", 0)
    return pf * wr * math.sqrt(tr + 1)          # rewards winrate × PF × MORE trades


class TFState:
    def __init__(self, tf, rr, pre):
        self.tf, self.rr, self.pre = tf, rr, pre
        self.seen = set(); self.elites = []; self.best = None
        self.evals = 0; self.dups = 0
        self.lock = threading.Lock()

    def claim(self, k):
        with self.lock:
            if k in self.seen:
                self.dups += 1; return False
            self.seen.add(k); return True

    def submit(self, g, s):
        sc = score(s)
        with self.lock:
            self.evals += 1
            if self.best is None or sc >= self.best["score"]:
                self.best = {"name": f"Agent-{self.tf}", "tf": self.tf, "rr": self.rr,
                             "trades": s["trades"], "winrate": s["winrate"], "pf": s["pf"],
                             "net_profit": s["net_profit"], "score": round(sc, 4),
                             "evals": self.evals, "dups": self.dups, "params": asdict(g),
                             "updated": time.strftime("%H:%M:%S")}
            else:
                self.best["evals"] = self.evals; self.best["dups"] = self.dups
            self.elites.append(g)
            if len(self.elites) > 40:
                self.elites = self.elites[-40:]

    def parents(self, rng):
        with self.lock:
            if len(self.elites) >= 2:
                return rng.sample(self.elites, 2)
            if self.elites:
                return [self.elites[-1], self.elites[-1]]
        return None


def worker(st, rng):
    while True:
        try:
            par = st.parents(rng)
            if par is None:
                g = mutate_unified(random_unified(rng), 0.4, rng)
            else:
                g = mutate_unified(crossover_unified(par[0], par[1], rng), 0.3, rng)
            g.rr_target = st.rr
            if not st.claim(g.key()):
                continue
            s = run_backtest(None, g, st.pre)     # shared precomp -> no per-agent memory
            st.submit(g, s)
        except Exception:
            time.sleep(1)


def main():
    threading.stack_size(256 * 1024)              # small stacks -> hundreds of threads OK
    df = load_data("data")
    total = len(TFS) * AGENTS_PER_TF
    print(f"[main] M1={len(df):,} bars | TFs={len(TFS)} | agents/TF={AGENTS_PER_TF} "
          f"| TOTAL AGENTS={total} (shared precomp -> no OOM)", flush=True)
    states = {}
    for tf, rule, rr in TFS:
        rdf = resample(df, rule)
        if MAX_BARS and len(rdf) > MAX_BARS:
            rdf = rdf.iloc[-MAX_BARS:].reset_index(drop=True)   # cap -> smaller precomp
        if len(rdf) < 300:
            continue
        pre = get_precomp(rdf); del rdf
        st = TFState(tf, rr, pre); states[tf] = st
        for w in range(AGENTS_PER_TF):
            threading.Thread(target=worker, args=(st, random.Random(1000 + w * 7 + (ord(tf[0]) % 97))),
                             daemon=True).start()
        print(f"[{tf}] precomp ready | {AGENTS_PER_TF} agents online", flush=True)
    del df  # free M1 frame (precomps hold their own arrays)
    print(f"[main] ALL {len(states)*AGENTS_PER_TF} agents running -> infinite backtesting loop", flush=True)

    from util import safe_write_json
    while True:
        time.sleep(10)
        out = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "total_agents": len(states) * AGENTS_PER_TF, "agents": {}, "total_evals": 0}
        te = 0
        for tf, st in states.items():
            if st.best:
                out["agents"][tf] = st.best
            te += st.evals
        out["total_evals"] = te
        try:
            safe_write_json(MEM, out)
        except Exception:
            pass
        print(f"[main] agents={out['total_agents']} | total_evals={te:,} | "
              f"TFs={len(out['agents'])}/{len(states)}", flush=True)


if __name__ == "__main__":
    main()
