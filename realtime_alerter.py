"""
realtime_alerter.py
===================
Live-feed loop that turns the evolution's consensus into PAPER trades + alerts.

  - PriceFeed: SimulatedFeed (deterministic walk off last known close) or
    cTraderFeed stub (requires creds; otherwise simulated).
  - Every TICK seconds: read consensus from the engine, manage one paper trade
    at a time (open on fresh buy/sell when flat, SL/TP resolved via PaperTracker
    SL-guard), optionally mirror to DemoTrader (demo, tiny lot), and push a
    Telegram alert if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set.
"""
from __future__ import annotations
import os, time, threading, random
import requests

SL_DEFAULT = 0.5
TP_DEFAULT = 2.5     # 1:5
LOT_DEMO = 0.01


# ----------------------------------------------------- price feeds
class PriceFeed:
    def price(self) -> float: ...           # noqa


class SimulatedFeed(PriceFeed):
    def __init__(self, start: float = 2000.0, seed: int = 1):
        self._p = start
        self._rng = random.Random(seed)

    def price(self) -> float:
        self._p += self._rng.gauss(0, 0.35)
        return round(max(100.0, self._p), 3)


class cTraderFeed(PriceFeed):       # pragma: no cover - needs creds
    """Stub: subscribe to cTrader OpenAPI spot stream for XAUUSD.
    Until wired, behaves like SimulatedFeed."""
    def __init__(self, start: float = 2000.0):
        self._sim = SimulatedFeed(start)

    def price(self) -> float:
        return self._sim.price()


def build_feed(last_close: float = 2000.0) -> PriceFeed:
    if os.environ.get("CTRADER_CLIENT_ID"):
        try:
            return cTraderFeed(last_close)
        except Exception:
            pass
    return SimulatedFeed(last_close)


# ----------------------------------------------------- telegram
class TelegramAlerter:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat = os.environ.get("TELEGRAM_CHAT_ID", "")

    def send(self, text: str):
        if not (self.token and self.chat):
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data={"chat_id": self.chat, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            return r.status_code == 200
        except Exception as e:
            print(f"[telegram] send failed: {e}")
            return False


# ----------------------------------------------------- main loop
class RealtimeAlerter:
    def __init__(self, consensus_fn, tracker, demo=None, feed=None, tick: int = 60):
        self.consensus_fn = consensus_fn
        self.tracker = tracker
        self.demo = demo
        self.feed = feed or build_feed()
        self.tick = tick
        self.tg = TelegramAlerter()
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        last_sig = "flat"
        print("[alerter] started (paper trading).")
        while not self._stop.is_set():
            try:
                px = self.feed.price()
                self.tracker.update(px)                 # resolve open paper trades
                cons = self.consensus_fn()
                sig = cons.get("signal", "flat")
                flat = self.tracker.stats()["open_count"] == 0

                if sig in ("buy", "sell") and flat and sig != last_sig:
                    side = 1 if sig == "buy" else -1
                    t = self.tracker.open_trade(side, px, SL_DEFAULT, TP_DEFAULT, LOT_DEMO)
                    last_sig = sig
                    msg = (f"*XAUUSD {sig.upper()}* (paper)\n"
                           f"entry ~{px}  SL {t['sl']}  TP {t['tp']}  lot {LOT_DEMO}\n"
                           f"consensus score {cons.get('score')} ({cons.get('agreement')} agree, {cons.get('voters')} agents)")
                    self.tg.send(msg)
                    if self.demo is not None:
                        try:
                            self.demo.place_market_order(side, LOT_DEMO, t["sl"], t["tp"], price_ref=px)
                        except Exception as e:
                            print(f"[alerter] demo order skipped: {e}")
                elif sig == "flat":
                    last_sig = "flat"
            except Exception as e:
                print(f"[alerter] loop error: {e}")
            self._stop.wait(self.tick)


def start_alerter(consensus_fn, tracker, demo=None, feed=None, tick: int = 60) -> RealtimeAlerter:
    al = RealtimeAlerter(consensus_fn, tracker, demo=demo, feed=feed, tick=tick)
    threading.Thread(target=al.run, daemon=True, name="realtime-alerter").start()
    return al


if __name__ == "__main__":
    from live_tracker import PaperTracker
    pt = PaperTracker("memory/paper_test.json")
    pt.open_trades.clear(); pt.closed.clear()
    al = RealtimeAlerter(lambda: {"signal": "buy", "score": 12, "agreement": 0.4, "voters": 12},
                         pt, feed=SimulatedFeed(2000.0), tick=1)
    t = threading.Thread(target=al.run, daemon=True)
    t.start(); time.sleep(6); al.stop(); t.join(timeout=3)
    print("paper stats:", pt.stats())
