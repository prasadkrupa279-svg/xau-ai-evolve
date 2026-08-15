"""
genome.py
=========
Agent genome + true dedup key + three-objective fitness.

Genome fields:
  mode, sl, tp, psw, pwk, pdp, ptr, sess, enabled(37-bit mask), ind_conf,
  sl_mode, tp_ratio, use_ppa, use_adv

Fitness optimises (simultaneously, never trading one for another):
  winrate^3  *  (profit_factor)^1.5  *  trades_factor  *  net_profit(mild)

The 50/50 SL split ($0.5 / $1.0) is enforced at the *population* level by the
evolution engine (half the genomes use sl=0.5, half sl=1.0), NOT in the fitness.
"""
from __future__ import annotations
import json, random, hashlib
from dataclasses import dataclass, asdict, field
import numpy as np

from indicators_lib import VOTE_DIM

SL_CHOICES = (0.5, 1.0)          # 50/50 enforced by engine
RR_DEFAULT = 5.0                 # 1:5 reward:risk
PTR_CHOICES = (0, 100, 200)      # 0 off, 100 EMA50&100, 200 EMA100&200


@dataclass
class Genome:
    mode: int = 0
    sl: float = 0.5
    tp: float = 2.5
    psw: int = 8            # swing lookback
    pwk: float = 0.30       # wick/zone tolerance ($)
    pdp: float = 0.50       # OB impulse move ($)
    ptr: int = 0            # trend filter
    sess: int = 0           # session filter (0=all)
    enabled: int = 0        # 37-bit mask
    ind_conf: int = 5       # min |net votes| to fire
    sl_mode: int = 0
    tp_ratio: float = 5.0
    use_ppa: bool = True
    use_adv: bool = False

    # ---- dedup: full-genome key
    def key(self) -> str:
        blob = json.dumps({
            "mode": self.mode, "sl": round(self.sl, 4), "tp": round(self.tp, 4),
            "psw": self.psw, "pwk": round(self.pwk, 4), "pdp": round(self.pdp, 4),
            "ptr": self.ptr, "sess": self.sess, "enabled": self.enabled,
            "ind_conf": self.ind_conf, "sl_mode": self.sl_mode,
            "tp_ratio": round(self.tp_ratio, 4), "use_ppa": self.use_ppa,
            "use_adv": self.use_adv,
        }, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:12]

    def mask_array(self) -> np.ndarray:
        b = np.zeros(VOTE_DIM, dtype=np.int8)
        for j in range(VOTE_DIM):
            if (self.enabled >> j) & 1:
                b[j] = 1
        if b.sum() == 0:                 # never a degenerate "vote nothing" agent
            b[: max(3, self.ind_conf)] = 1
        return b

    def to_dict(self): return asdict(self)


# --------------------------------------------------------------- generation
def random_genome(sl: float | None = None, rng: random.Random | None = None) -> Genome:
    rng = rng or random
    sl = sl if sl is not None else rng.choice(SL_CHOICES)
    rr = RR_DEFAULT
    enabled = 0
    k = rng.randint(6, VOTE_DIM)              # 6..37 enabled sources
    bits = rng.sample(range(VOTE_DIM), k)
    for b in bits:
        enabled |= (1 << b)
    return Genome(
        mode=rng.choice([0, 1]),
        sl=sl, tp=round(sl * rr, 2),
        psw=rng.choice([5, 8, 10, 13, 20]),
        pwk=round(rng.uniform(0.15, 0.60), 2),
        pdp=round(rng.uniform(0.30, 0.90), 2),
        ptr=rng.choice(PTR_CHOICES),
        sess=rng.choice([0, 1, 2, 3]),
        enabled=enabled,
        ind_conf=rng.randint(2, 12),
        sl_mode=rng.choice([0, 1]),
        tp_ratio=rr,
        use_ppa=True, use_adv=rng.random() < 0.3,
    )


def mutate(g: Genome, rate: float = 0.25, rng: random.Random | None = None) -> Genome:
    rng = rng or random
    d = g.to_dict()
    if rng.random() < rate:
        d["psw"] = rng.choice([5, 8, 10, 13, 20])
    if rng.random() < rate:
        d["pwk"] = round(rng.uniform(0.15, 0.60), 2)
    if rng.random() < rate:
        d["pdp"] = round(rng.uniform(0.30, 0.90), 2)
    if rng.random() < rate:
        d["ptr"] = rng.choice(PTR_CHOICES)
    if rng.random() < rate:
        d["ind_conf"] = max(1, min(VOTE_DIM, d["ind_conf"] + rng.choice([-2, -1, 1, 2])))
    if rng.random() < rate * 0.8:
        mask = d["enabled"]
        for _ in range(rng.randint(1, 4)):
            mask ^= (1 << rng.randrange(VOTE_DIM))
        d["enabled"] = mask
    # sl/rr are NOT mutated: engine owns the 50/50 split
    return Genome(**d)


def crossover(a: Genome, b: Genome, rng: random.Random | None = None) -> Genome:
    """Uniform per-gene crossover of two parents (keeps each parent's sl)."""
    rng = rng or random
    da, db = a.to_dict(), b.to_dict()
    child = {}
    for k, va in da.items():
        child[k] = va if rng.random() < 0.5 else db[k]
    # blend the 37-bit mask gene-by-gene
    m = 0
    for j in range(VOTE_DIM):
        bit = ((a.enabled >> j) & 1) if rng.random() < 0.5 else ((b.enabled >> j) & 1)
        m |= (bit << j)
    child["enabled"] = m
    return Genome(**child)


# --------------------------------------------------------------- fitness
def fitness(stats: dict) -> float:
    """
    stats keys: trades, wins, losses, winrate(0..1), pf(>=0), net_profit($).
    All four objectives multiplied so none can be sacrificed for another.
    """
    trades = stats.get("trades", 0)
    wr = stats.get("winrate", 0.0)
    pf = stats.get("pf", 0.0)
    net = stats.get("net_profit", 0.0)

    if trades < 30:                       # statistically meaningless
        return 0.0

    f_wr = max(0.0, wr) ** 3              # winrate always up
    f_pf = max(0.0, pf) ** 1.5            # profit factor

    # trades factor: ramps to 1.0 by 1000, plateaus to 3000, mild decay after
    if trades < 1000:
        f_tr = 0.25 + 0.75 * (trades / 1000.0)
    elif trades <= 3000:
        f_tr = 1.0
    else:
        f_tr = max(0.4, 1.0 - (trades - 3000) / 15000.0)

    # net profit mild (only matters when everything else is close)
    f_net = 1.0 + 0.25 * np.tanh(net / 1000.0)

    return float(f_wr * f_pf * f_tr * f_net)


def stats_from(wins: int, losses: int, timeouts: int, gross_win: float, gross_loss: float, net: float) -> dict:
    trades = wins + losses + timeouts
    winrate = wins / trades if trades else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
    return {
        "trades": trades, "wins": wins, "losses": losses, "timeouts": timeouts,
        "winrate": round(winrate, 4), "pf": round(pf, 4),
        "net_profit": round(net, 2),
    }


if __name__ == "__main__":
    rng = random.Random(0)
    a, b = random_genome(0.5, rng), random_genome(1.0, rng)
    c = crossover(a, b, rng)
    print("A key", a.key(), "sl", a.sl, "| B key", b.key(), "sl", b.sl, "| C key", c.key())
    s = stats_from(330, 670, 0, 330 * 2.5, 670 * 0.5, 330 * 2.5 - 670 * 0.5)
    print("example honest stats @1:5:", s, "-> fitness", round(fitness(s), 4))
