"""
optimize_unified.py
===================
Evolutionary optimiser for the Arena Unified Liquidity strategy on REAL data.
Maximises winrate^3 x PF^1.5 x trades-factor (all three up at once).
Prints baseline -> per-gen champion -> final improved params (for the Pine).
"""
from __future__ import annotations
import json, os, time, random
from data_tools import load_data
from strategy_unified import (get_precomp, run_backtest, fitness,
                              UnifiedGenome, random_unified, crossover_unified, mutate_unified)

POP = 40
GENS = 20
ELITE = 6
MEMORY = "memory/unified_champion.json"


def main():
    rng = random.Random(7)
    df = load_data("data")
    pre = get_precomp(df)
    print(f"data: {len(df):,} bars | pop={POP} gens={GENS}")

    # baseline = Pine defaults
    base = run_backtest(df, UnifiedGenome(), pre)
    print(f"BASELINE (Pine defaults): trades={base['trades']} wr={base['winrate']} "
          f"pf={base['pf']} net=${base['net_profit']} fit={fitness(base):.4f}")

    # init population (seed a few near-default + a TZ-offset sweep so it bootstraps)
    pop = [random_unified(rng) for _ in range(POP)]
    seen = {g.key() for g in pop}
    best_ever = base
    stag = 0
    mut = 0.3

    for gen in range(1, GENS + 1):
        scored = []
        for g in pop:
            s = run_backtest(df, g, pre)
            scored.append((g, s, fitness(s)))
        scored.sort(key=lambda x: x[2], reverse=True)

        champ = scored[0][1]
        if fitness(champ) > fitness(best_ever) + 1e-9:
            best_ever = champ; stag = 0; mut = max(0.1, mut - 0.02)
        else:
            stag += 1
            if stag >= 4:
                mut = min(0.55, mut + 0.06); stag = 0

        elites = [g for g, _, _ in scored[:ELITE]]
        nxt = list(elites)
        while len(nxt) < POP:
            parents = rng.sample(elites, 2) if len(elites) >= 2 else elites
            child = crossover_unified(parents[0], parents[1], rng)
            child = mutate_unified(child, mut, rng)
            t = 0
            while child.key() in seen and t < 30:
                child = mutate_unified(child, mut + 0.1, rng); t += 1
            seen.add(child.key())
            nxt.append(child)
        pop = nxt

        print(f"gen{gen:>2} | best_fit={fitness(champ):.4f} mut={mut:.2f} stag={stag} | "
              f"champ trades={champ['trades']} wr={champ['winrate']} pf={champ['pf']} "
              f"net=${champ['net_profit']}")

    print("\n================ BEST EVER ================")
    print(f"trades={best_ever['trades']} wr={best_ever['winrate']} pf={best_ever['pf']} "
          f"net=${best_ever['net_profit']} fit={fitness(best_ever):.4f}")
    p = best_ever["params"]
    print("params:", json.dumps(p, indent=0))
    os.makedirs("memory", exist_ok=True)
    with open(MEMORY, "w") as f:
        json.dump({"baseline": base, "best": best_ever, "bars": len(df)}, f, indent=2)
    print(f"\nsaved -> {MEMORY}")

    # improvement deltas
    db = base; bb = best_ever
    print(f"\nDELTA vs baseline: trades {db['trades']}->{bb['trades']} "
          f"({(bb['trades']/max(1,db['trades'])-1)*100:+.0f}%) | "
          f"wr {db['winrate']*100:.1f}%->{bb['winrate']*100:.1f}% | "
          f"pf {db['pf']:.2f}->{bb['pf']:.2f} | net ${db['net_profit']}->${bb['net_profit']}")


if __name__ == "__main__":
    main()
