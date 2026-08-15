"""
ai_agent_daemon.py
==================
The 24/7 evolution daemon. Computes votes ONCE, then runs generations forever,
persisting champion + leaderboard to GLOBAL_AI_MEMORY (restart-safe).

Runnable standalone, or imported and launched in a background thread by the
dashboard (so a single Render web service serves UI + keeps evolving).
"""
from __future__ import annotations
import os, time, threading, traceback

DATA_DIR = os.environ.get("DATA_DIR", "data")
MEMORY_PATH = os.environ.get("MEMORY_PATH", "memory/global_ai_memory.json")
SLEEP_BETWEEN_GENS = float(os.environ.get("GEN_SLEEP", "0.2"))

# live process state (read by the dashboard)
STATE = {
    "running": False, "gen": 0, "data_source": None, "bars": 0,
    "last_gen": None, "started_at": None, "error": None,
    "best_fit": 0.0,
}
_lock = threading.Lock()


def _log(msg):
    print(f"[daemon] {msg}", flush=True)


def run_daemon(generations: int | None = None, data_dir: str = DATA_DIR,
               memory_path: str = MEMORY_PATH, sleep: float = SLEEP_BETWEEN_GENS,
               log=_log):
    from data_tools import ensure_data
    from indicators_lib import compute_votes
    from evolution import EvolutionEngine

    with _lock:
        STATE["running"] = True
        STATE["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE["error"] = None

    try:
        df, src = ensure_data(data_dir)
        votes, names = compute_votes(df)
        with _lock:
            STATE["data_source"] = src
            STATE["bars"] = len(df)
        log(f"data={src} bars={len(df):,}  votes={votes.shape}  -> evolving")

        eng = EvolutionEngine(df, votes, memory_path=memory_path)

        g = 0
        consecutive_err = 0
        while generations is None or g < generations:
            try:
                r = eng.step()
                consecutive_err = 0
                with _lock:
                    STATE["error"] = None
            except Exception as e:
                consecutive_err += 1
                with _lock:
                    STATE["error"] = f"gen error #{consecutive_err}: {e}"
                log(f"gen error #{consecutive_err}: {e}\n{traceback.format_exc()}")
                # back off, then keep going (never give up). Re-init population if it got wiped.
                time.sleep(min(30, 2 ** consecutive_err))
                if consecutive_err % 20 == 0:
                    log("too many errors -> re-initialising population")
                    try:
                        eng._init_population()
                    except Exception:
                        pass
                continue
            c = r["champion"]
            with _lock:
                STATE["gen"] = r["gen"]
                STATE["last_gen"] = r
                STATE["best_fit"] = r["best_fit"]
            log(f"gen {r['gen']:>4} | fit={r['best_fit']} mut={r['mut_rate']} stag={r['stagnant']} | "
                f"champ sl={c['sl']} trades={c['trades']} wr={c['winrate']} pf={c['pf']} net=${c['net_profit']} ({r['took_s']}s)")
            g += 1
            if sleep > 0:
                time.sleep(sleep)
    except Exception as e:
        with _lock:
            STATE["error"] = f"{e}"
        log(f"FATAL: {e}\n{traceback.format_exc()}")
    finally:
        with _lock:
            STATE["running"] = False


def start_background(**kw) -> threading.Thread:
    t = threading.Thread(target=run_daemon, kwargs=kw, daemon=True, name="evolution-daemon")
    t.start()
    return t


def snapshot() -> dict:
    with _lock:
        return dict(STATE)


if __name__ == "__main__":
    # run a few gens locally for a sanity check
    run_daemon(generations=int(os.environ.get("GENS", "6")))
