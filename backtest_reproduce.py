"""
backtest_reproduce.py
=====================
Standalone, fully reproducible backtest. No evolution, no daemon — just:
load data -> compute 37 indicators -> backtest ONE agent (champion from memory
or a random/specified genome) -> print HONEST stats + the agent's enabled
indicator set.

Usage:
    python backtest_reproduce.py                      # uses champion from memory/
    python backtest_reproduce.py --sl 0.5 --seed 7    # random agent, sl=0.5
    python backtest_reproduce.py --bars 200000        # cap bars for speed
"""
from __future__ import annotations
import argparse, json, os, random, time

from data_tools import ensure_data
from indicators_lib import compute_votes, VOTE_NAMES
from genome import Genome, random_genome, fitness
from ppa_engine import run_backtest, CONTRACT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--memory", default="memory/global_ai_memory.json")
    ap.add_argument("--sl", type=float, default=None, choices=[None, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=None, help="random agent seed")
    ap.add_argument("--bars", type=int, default=0, help="cap bars (0=all)")
    ap.add_argument("--lot", type=float, default=1.0)
    args = ap.parse_args()

    df, src = ensure_data(args.data)
    if args.bars and len(df) > args.bars:
        df = df.iloc[-args.bars:].reset_index(drop=True)
    print(f"== data: {src}  bars={len(df):,}  contract={CONTRACT} oz/lot  lot={args.lot}")

    votes, _ = compute_votes(df)

    if args.seed is not None:
        g = random_genome(args.sl, random.Random(args.seed))
        print(f"== random agent (seed={args.seed})")
    elif os.path.exists(args.memory):
        with open(args.memory) as f:
            mem = json.load(f)
        g = Genome(**mem["champion"]["genome"])
        print(f"== champion from {args.memory} (gen {mem.get('gen')})")
    else:
        g = random_genome(args.sl or 0.5)
        print("== random agent (no memory found)")

    print(f"== genome key={g.key()}  sl=${g.sl} tp=${g.tp} rr=1:{g.tp/g.sl:.0f}  "
          f"ptr={g.ptr} psw={g.psw} pwk={g.pwk} ind_conf={g.ind_conf}  "
          f"enabled={bin(g.enabled).count('1')}/37")

    enabled = [VOTE_NAMES[j] for j in range(37) if (g.enabled >> j) & 1]
    print("== enabled vote-sources:", ", ".join(enabled))

    t0 = time.time()
    s = run_backtest(df, votes, g, lot=args.lot)
    dt = time.time() - t0

    print("\n================ HONEST BACKTEST RESULT ================")
    print(f" trades        : {s['trades']:>6}   (wins {s['wins']} / losses {s['losses']} / timeout {s['timeouts']})")
    print(f" winrate       : {s['winrate']*100:>6.2f} %")
    print(f" profit factor : {s['pf']:>6.3f}")
    print(f" net profit    : ${s['net_profit']:>,.2f}   (lot {args.lot})")
    print(f" fitness       : {fitness(s):.5f}")
    print(f" backtest time : {dt:.2f}s")
    print("=======================================================")
    print("NOTE: SL-guard ON (same-bar SL beats TP). 1:5 RR, no trailing/partials.")
    if s["winrate"] > 0.45 and s["pf"] > 3 and src == "synthetic":
        print("WARN: suspiciously good on synthetic data - recheck for look-ahead.")


if __name__ == "__main__":
    main()
