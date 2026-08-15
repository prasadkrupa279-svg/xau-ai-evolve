"""
evolution.py
============
Quantum-inspired genetic evolution of trading agents.

  - population of POP_SIZE unique genomes (true dedup via Genome.key)
  - 50/50 SL split: half sl=$0.5, half sl=$1.0  (engine-enforced, not fitness)
  - elitism: top survivors carried over unchanged
  - ADAPTIVE mutation: if best fitness stagnant >= STAG_GENS -> mutation rate rises
    (explore); while improving -> mutation rate falls (exploit). Never stuck.
  - GLOBAL_AI_MEMORY: champion + leaderboard + gen counter persisted to JSON
    (restart-safe).
  - consensus signal: top-K agents vote on the latest closed bar -> buy/sell.
"""
from __future__ import annotations
import json, os, time, random
from dataclasses import asdict
import numpy as np
import pandas as pd

from genome import Genome, random_genome, crossover, mutate, fitness, RR_DEFAULT, SL_CHOICES
from ppa_engine import run_backtest

POP_SIZE = 50
ELITE = 8
STAG_GENS = 5
MUT_MIN, MUT_MAX = 0.08, 0.55
TOP_K_CONSENSUS = 12


class EvolutionEngine:
    def __init__(self, df: pd.DataFrame, votes: np.ndarray,
                 memory_path: str = "memory/global_ai_memory.json",
                 pop_size: int = POP_SIZE, seed: int = 1):
        self.df = df
        self.votes = votes
        self.memory_path = memory_path
        self.pop_size = pop_size
        self.rng = random.Random(seed)
        self.gen = 0
        self.best_fit = 0.0
        self.stagnant = 0
        self.mut_rate = 0.25
        self.pop: list[Genome] = []
        self.scores: dict[str, dict] = {}          # key -> {genome, stats, fit}
        self.used_keys: set[str] = set()
        self._init_population()

    # ----------------------------------------------------------- population
    def _make_unique(self, sl: float | None = None, max_tries: int = 200) -> Genome:
        for _ in range(max_tries):
            g = random_genome(sl, self.rng)
            if g.key() not in self.used_keys:
                self.used_keys.add(g.key())
                return g
        # fall back to a mutated random to force uniqueness
        g = random_genome(sl, self.rng)
        self.used_keys.add(g.key())
        return g

    def _init_population(self):
        # 50/50 sl split
        half = self.pop_size // 2
        self.pop = [self._make_unique(0.5) for _ in range(half)] + \
                   [self._make_unique(1.0) for _ in range(self.pop_size - half)]

    def _evaluate(self, g: Genome) -> dict:
        if g.key() in self.scores:
            return self.scores[g.key()]
        stats = run_backtest(self.df, self.votes, g)
        rec = {"genome": asdict(g), "key": g.key(), "stats": stats, "fit": fitness(stats)}
        self.scores[g.key()] = rec
        return rec

    def _breed_child(self, parents, sl_forced: float | None = None) -> Genome:
        a, b = self.rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
        child = crossover(a, b, self.rng)
        child = mutate(child, self.mut_rate, self.rng)
        if sl_forced is not None:
            child.sl = sl_forced
            child.tp = round(sl_forced * RR_DEFAULT, 2)
            child.tp_ratio = RR_DEFAULT
        # ensure uniqueness
        tries = 0
        while child.key() in self.used_keys and tries < 100:
            child = mutate(child, self.mut_rate + 0.1, self.rng)
            if sl_forced is not None:
                child.sl = sl_forced; child.tp = round(sl_forced * RR_DEFAULT, 2)
            tries += 1
        self.used_keys.add(child.key())
        return child

    # ----------------------------------------------------------- one generation
    def step(self) -> dict:
        t0 = time.time()
        # 1. evaluate whole population
        recs = [self._evaluate(g) for g in self.pop]

        # 2. rank by fitness (desc); ties broken by trades then pf
        recs.sort(key=lambda r: (r["fit"], r["stats"]["trades"], r["stats"]["pf"]), reverse=True)
        elite_genomes = [Genome(**r["genome"]) for r in recs[:ELITE]]

        # 3. adaptive mutation based on stagnation
        cur_best = recs[0]["fit"]
        if cur_best > self.best_fit + 1e-9:
            self.best_fit = cur_best
            self.stagnant = 0
            self.mut_rate = max(MUT_MIN, self.mut_rate - 0.03)
        else:
            self.stagnant += 1
            if self.stagnant >= STAG_GENS:
                self.mut_rate = min(MUT_MAX, self.mut_rate + 0.08)
                self.stagnant = 0   # reset so it keeps exploring

        # 4. build next gen: elites (kept) + bred children, 50/50 sl split
        next_pop: list[Genome] = list(elite_genomes)
        # normalise elite sl distribution a touch, then fill 50/50
        n_half = self.pop_size // 2
        cur_05 = sum(1 for g in next_pop if abs(g.sl - 0.5) < 1e-6)
        need_05 = max(0, n_half - cur_05)
        while len(next_pop) < self.pop_size:
            sl_forced = 0.5 if need_05 > 0 else 1.0
            if need_05 > 0:
                need_05 -= 1
            next_pop.append(self._breed_child(elite_genomes, sl_forced))
        # ensure exact 50/50 by overriding any drift
        self._force_split(next_pop)
        self.pop = next_pop[: self.pop_size]

        self.gen += 1
        champ = recs[0]
        self._save_memory(champ, recs)
        return {
            "gen": self.gen, "best_fit": round(self.best_fit, 5), "mut_rate": round(self.mut_rate, 3),
            "stagnant": self.stagnant,
            "champion": {**champ["stats"], "key": champ["key"]},
            "took_s": round(time.time() - t0, 2),
        }

    def _force_split(self, pop: list[Genome]):
        idx05 = [i for i, g in enumerate(pop) if abs(g.sl - 0.5) < 1e-6]
        idx10 = [i for i, g in enumerate(pop) if abs(g.sl - 1.0) < 1e-6]
        target = len(pop) // 2
        # if too many 0.5, flip extras to 1.0
        while len(idx05) > target and idx10:
            # move a non-elite 0.5 to 1.0
            i = idx05.pop(-1)
            pop[i].sl = 1.0; pop[i].tp = round(1.0 * RR_DEFAULT, 2)
            idx10.append(i)
        while len(idx10) > target and idx05:
            i = idx10.pop(-1)
            pop[i].sl = 0.5; pop[i].tp = round(0.5 * RR_DEFAULT, 2)
            idx05.append(i)

    # ----------------------------------------------------------- persistence
    def _save_memory(self, champ, recs):
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        leaderboard = [{
            "key": r["key"], "sl": r["stats"]["sl"], "rr": r["stats"]["rr"],
            "trades": r["stats"]["trades"], "winrate": r["stats"]["winrate"],
            "pf": r["stats"]["pf"], "net": r["stats"]["net_profit"], "fit": round(r["fit"], 5),
            "genome": r["genome"],
        } for r in recs[: self.pop_size]]
        data = {
            "gen": self.gen, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_fit": round(self.best_fit, 5), "mut_rate": round(self.mut_rate, 3),
            "stagnant": self.stagnant,
            "champion": {**champ["stats"], "key": champ["key"], "genome": champ["genome"]},
            "leaderboard": leaderboard,
        }
        tmp = self.memory_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.memory_path)

    # ----------------------------------------------------------- consensus signal
    def consensus_signal(self) -> dict:
        """Top-K agents vote on the LAST closed bar -> net direction."""
        if not os.path.exists(self.memory_path):
            return {"signal": "flat", "agreement": 0.0, "votes": 0}
        with open(self.memory_path) as f:
            mem = json.load(f)
        top = mem.get("leaderboard", [])[:TOP_K_CONSENSUS]
        if not top:
            return {"signal": "flat", "agreement": 0.0, "votes": 0}
        last_votes = self.votes[-1]
        tally = 0
        for a in top:
            g = Genome(**a["genome"])
            m = g.mask_array()
            tally += int((last_votes * m).sum())
        if tally > 0:
            sig = "buy"
        elif tally < 0:
            sig = "sell"
        else:
            sig = "flat"
        agree = abs(tally) / max(1, sum(a["genome"]["enabled"].bit_count() for a in top))
        return {"signal": sig, "score": int(tally), "agreement": round(min(1.0, agree), 3),
                "voters": len(top)}

    # ----------------------------------------------------------- leaderboard accessor
    def leaderboard(self) -> list[dict]:
        if not os.path.exists(self.memory_path):
            return []
        with open(self.memory_path) as f:
            return json.load(f).get("leaderboard", [])


if __name__ == "__main__":
    from data_tools import ensure_data
    from indicators_lib import compute_votes
    df, src = ensure_data()
    v, _ = compute_votes(df)
    print(f"data={src} bars={len(df):,}  -> warming evolution (5 gens)...")
    eng = EvolutionEngine(df, v, memory_path="memory/test_memory.json")
    for i in range(5):
        r = eng.step()
        c = r["champion"]
        print(f"  gen{r['gen']:>2} best_fit={r['best_fit']} mut={r['mut_rate']} "
              f"| champ sl={c['sl']} trades={c['trades']} wr={c['winrate']} pf={c['pf']} net=${c['net_profit']} ({r['took_s']}s)")
    print("consensus:", eng.consensus_signal())
    # verify 50/50 split in leaderboard
    lb = eng.leaderboard()
    n05 = sum(1 for a in lb if abs(a["sl"] - 0.5) < 1e-6)
    print(f"leaderboard size={len(lb)} | sl0.5={n05} sl1.0={len(lb)-n05}")
