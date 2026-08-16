"""
multi_agent_daemon.py — PARALLEL multi-agent infinite-loop strategy optimizer.

Design (exactly as requested):
  - N agents, each OWNS a different gene-focus -> har agent alag kaam karta hai.
  - GLOBAL dedup registry (shared evaluated-set) -> koi bhi agent kisi ka kaam
    repeat nahi karta (genome key already evaluated by anyone = skip).
  - RR PINNED (unchanged). Agents only tune the OTHER genes.
  - Fitness rewards winrate + trades (+PF) -> dono upar.
  - Shared elite pool + global best -> agents build on each other.
  - Infinite loop per agent (24/7 style, never stops).
"""
from __future__ import annotations
import os, time, json, random, threading, traceback
from dataclasses import asdict
from data_tools import load_data
from strategy_unified import (get_precomp, run_backtest, fitness, RANGES,
                              UnifiedGenome, random_unified, crossover_unified)

RR_FIXED = 3.0          # 1m RR pinned (user: "RR wahi rehne de")
N_AGENTS = 6
MEM = "memory/multi_agent_best.json"

FOCUSES = {
    "A-Sessions": ["asian_start", "asian_end", "sweep_start", "sweep_end", "confirm_end"],
    "B-Confirm":  ["confirm_mode", "min_body", "close_pos_min", "range_atr"],
    "C-Stops":    ["min_stop_atr", "stop_buffer"],
    "D-Trend":    ["use_daily", "use_h1", "use_h4"],
    "E-Timing":   ["bars_after_sweep", "cooldown", "entry_hour_start", "entry_hour_end"],
    "F-Explorer": (list(RANGES.keys()) + ["use_daily", "use_h1", "use_h4",
                   "confirm_mode", "entry_hour_start", "entry_hour_end"]),
}


def mutate_focus(g, focus, rate, rng):
    d = asdict(g)
    d["rr_target"] = RR_FIXED
    for k in focus:
        if k in RANGES:
            lo, hi = RANGES[k]
            if rng.random() < rate:
                step = (hi - lo) * 0.15
                nv = max(lo, min(hi, d[k] + rng.uniform(-step, step)))
                d[k] = int(round(nv)) if isinstance(d[k], int) else round(nv, 2)
        elif k in ("use_daily", "use_h1", "use_h4"):
            if rng.random() < rate * 0.6:
                d[k] = not d[k]
        elif k == "confirm_mode":
            if rng.random() < rate * 0.6:
                d[k] = rng.randint(0, 1)
        elif k in ("entry_hour_start", "entry_hour_end"):
            if rng.random() < rate * 0.5:
                st = rng.randint(0, 23); d["entry_hour_start"] = st
                d["entry_hour_end"] = (st + rng.randint(1, 6)) % 24
    # tiny global jitter so Explorer + cross-focus recombination stay alive
    if rng.random() < 0.15:
        for k in RANGES:
            if rng.random() < 0.08:
                lo, hi = RANGES[k]
                d[k] = (rng.randint(int(lo), int(hi)) if isinstance(d[k], int)
                        else round(rng.uniform(lo, hi), 2))
    return UnifiedGenome(**d)


class Hub:
    def __init__(self, df, pre):
        self.df, self.pre = df, pre
        self.evaluated = set()          # GLOBAL dedup: no agent repeats another's work
        self.best = None
        self.elites = []
        self.lock = threading.Lock()
        self.evals = {k: 0 for k in FOCUSES}
        self.dups = {k: 0 for k in FOCUSES}

    def claim(self, key):                # atomic dedup
        with self.lock:
            if key in self.evaluated:
                return False
            self.evaluated.add(key)
            return True

    def submit(self, g, s):
        s["fit"] = fitness(s)
        with self.lock:
            if self.best is None or s["fit"] > self.best["fit"]:
                self.best = s
                print(f"[HUB] ★ NEW BEST fit={s['fit']:.4f} trades={s['trades']} "
                      f"wr={s['winrate']} pf={s['pf']} net=${s['net_profit']}", flush=True)
            self.elites.append(g)
            if len(self.elites) > 24:
                self.elites = self.elites[-24:]

    def parents(self, rng):
        with self.lock:
            if len(self.elites) >= 2:
                return rng.sample(self.elites, 2)
            if self.elites:
                return [self.elites[-1], self.elites[-1]]
        return None


def agent(name, focus, hub, seed):
    rng = random.Random(seed)
    print(f"[{name}] online | focus={focus[:3]}... ({len(focus)} genes) | RR pinned", flush=True)
    while True:
        try:
            par = hub.parents(rng)
            if par is None:
                g = mutate_focus(random_unified(rng), focus, 0.4, rng)
            else:
                g = mutate_focus(crossover_unified(par[0], par[1], rng), focus, 0.3, rng)
            g.rr_target = RR_FIXED
            if not hub.claim(g.key()):
                hub.dups[name] += 1
                continue
            hub.evals[name] += 1
            s = run_backtest(hub.df, g, hub.pre)
            hub.submit(g, s)
            if hub.evals[name] % 4 == 0:
                b = hub.best
                bf = f"{b['fit']:.4f}" if b else "0"
                bt = b["trades"] if b else 0
                bw = f"{b['winrate']*100:.1f}%" if b else "0%"
                print(f"[{name}] evals={hub.evals[name]} dups={hub.dups[name]} | "
                      f"global_best fit={bf} trades={bt} wr={bw}", flush=True)
        except Exception as e:
            print(f"[{name}] err: {e}", flush=True)
            time.sleep(2)


def main():
    df = load_data("data")
    pre = get_precomp(df)
    print(f"[main] data={len(df):,} bars | RR PINNED={RR_FIXED} | agents={N_AGENTS} "
          f"| GLOBAL dedup ON -> infinite multi-agent loop", flush=True)
    hub = Hub(df, pre)
    for i, (name, focus) in enumerate(FOCUSES.items()):
        threading.Thread(target=agent, args=(name, focus, hub, 100 + i), daemon=True).start()
    while True:
        time.sleep(10)
        if hub.best:
            try:
                from util import safe_write_json
                safe_write_json(MEM, {"best": hub.best, "total_evaluated": len(hub.evaluated),
                                      "per_agent_evals": hub.evals, "per_agent_dups": hub.dups,
                                      "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
            except Exception:
                pass
            b = hub.best
            print(f"[main] total_unique_genomes={len(hub.evaluated):,} | "
                  f"best fit={b['fit']:.4f} trades={b['trades']} wr={b['winrate']} "
                  f"pf={b['pf']} net=${b['net_profit']} | per-agent evals={hub.evals}", flush=True)


if __name__ == "__main__":
    main()
