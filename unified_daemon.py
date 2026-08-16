"""
unified_daemon.py — INFINITE-LOOP backtesting daemon (as specced: "infinite while loop").
Runs the genetic evolution of Arena-Unified-style agents on the REAL 1.06M M1 data,
forever, saving the champion each generation. 24/7-style never-stops loop.
"""
from __future__ import annotations
import os, time, json, random, traceback
from data_tools import load_data
from strategy_unified import (get_precomp, run_backtest, fitness,
                              UnifiedGenome, random_unified, crossover_unified, mutate_unified)

POP = int(os.environ.get("POP", "40"))
ELITE = 6
MEM = os.environ.get("MEMORY_PATH", "memory/unified_champion.json")
SLEEP = float(os.environ.get("GEN_SLEEP", "0"))


def main():
    rng = random.Random(int(os.environ.get("SEED", "7")))
    df = load_data("data")
    pre = get_precomp(df)
    print(f"[daemon] data={len(df):,} bars | POP={POP} -> INFINITE evolution loop starting", flush=True)

    pop = [random_unified(rng) for _ in range(POP)]
    seen = {g.key() for g in pop}
    best = None
    stag = 0
    mut = 0.3
    gen = 0

    while True:                                   # <-- infinite loop, as specced
        gen += 1
        t0 = time.time()
        try:
            scored = []
            for g in pop:
                s = run_backtest(df, g, pre)
                s["fit"] = fitness(s)
                scored.append((g, s))
            scored.sort(key=lambda x: x[1]["fit"], reverse=True)
            champ = scored[0][1]

            improved = best is None or champ["fit"] > best["fit"] + 1e-9
            if improved:
                best = champ; stag = 0; mut = max(0.1, mut - 0.02)
            else:
                stag += 1
                if stag >= 4:
                    mut = min(0.55, mut + 0.06); stag = 0

            elites = [g for g, _ in scored[:ELITE]]
            nxt = list(elites)
            while len(nxt) < POP:
                a, b = rng.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
                child = mutate_unified(crossover_unified(a, b, rng), mut, rng)
                tries = 0
                while child.key() in seen and tries < 30:
                    child = mutate_unified(child, mut + 0.1, rng); tries += 1
                seen.add(child.key())
                nxt.append(child)
            pop = nxt

            c = champ
            print(f"[daemon] gen {gen:>4} | fit={c['fit']:.4f} mut={mut:.2f} stag={stag} | "
                  f"champ trades={c['trades']} wr={c['winrate']} pf={c['pf']} net=${c['net_profit']} "
                  f"({time.time()-t0:.1f}s) | best_fit={best['fit']:.4f}", flush=True)

            # persist champion (restart-safe)
            try:
                from util import safe_write_json
                safe_write_json(MEM, {"gen": gen, "best": best, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "bars": len(df)})
            except Exception:
                pass
        except Exception as e:
            print(f"[daemon] gen {gen} error (continuing): {e}\n{traceback.format_exc()}", flush=True)
            time.sleep(5)
            continue
        if SLEEP > 0:
            time.sleep(SLEEP)


if __name__ == "__main__":
    main()
