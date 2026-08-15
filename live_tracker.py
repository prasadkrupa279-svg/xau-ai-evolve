"""
live_tracker.py
===============
Paper-trading result tracker. Opens virtual SL/TP trades on live feed prints,
resolves them with the SAME SL-guard rule as the backtester, and persists
results to JSON so the dashboard can show REAL paper performance
(not backtested, not faked).
"""
from __future__ import annotations
import json, os, time, uuid

CONTRACT = 100.0


class PaperTracker:
    def __init__(self, path: str = "memory/paper_results.json"):
        self.path = path
        self.open_trades: list[dict] = []
        self.closed: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    d = json.load(f)
                self.open_trades = d.get("open", [])
                self.closed = d.get("closed", [])
            except Exception:
                self.open_trades, self.closed = [], []

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"open": self.open_trades, "closed": self.closed[-500:]}, f)
        os.replace(tmp, self.path)

    def open_trade(self, side: int, price: float, sl: float, tp: float,
                   lot: float = 0.01, symbol: str = "XAUUSD") -> dict:
        if side > 0:
            tp_price, sl_price = price + tp, price - sl
        else:
            tp_price, sl_price = price - tp, price + sl
        t = {
            "id": uuid.uuid4().hex[:8], "symbol": symbol, "side": "buy" if side > 0 else "sell",
            "entry": round(price, 3), "sl": round(sl_price, 3), "tp": round(tp_price, 3),
            "lot": lot, "opened_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.open_trades.append(t)
        self._save()
        return t

    def update(self, price: float):
        """SL-guard resolution on the latest price tick."""
        still_open = []
        for t in self.open_trades:
            if t["side"] == "buy":
                sl_hit = price <= t["sl"]
                tp_hit = price >= t["tp"]
            else:
                sl_hit = price >= t["sl"]
                tp_hit = price <= t["tp"]
            if sl_hit and tp_hit:                # SL-guard: SL wins
                self._close(t, "loss", price)
            elif sl_hit:
                self._close(t, "loss", t["sl"])
            elif tp_hit:
                self._close(t, "win", t["tp"])
            else:
                still_open.append(t)
        self.open_trades = still_open
        self._save()

    def _close(self, t: dict, result: str, exit_price: float):
        side = 1 if t["side"] == "buy" else -1
        pnl = (exit_price - t["entry"]) * side * CONTRACT * t["lot"]
        t.update({"result": result, "exit": round(exit_price, 3), "pnl": round(pnl, 2),
                  "closed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.closed.append(t)

    def stats(self) -> dict:
        wins = [c for c in self.closed if c["result"] == "win"]
        losses = [c for c in self.closed if c["result"] == "loss"]
        gw = sum(c["pnl"] for c in wins)
        gl = abs(sum(c["pnl"] for c in losses))
        n = len(self.closed)
        wr = len(wins) / n if n else 0.0
        pf = gw / gl if gl > 0 else (10.0 if gw > 0 else 0.0)
        net = sum(c["pnl"] for c in self.closed)
        open_pnl = 0.0
        for t in self.open_trades:
            # mark-to-market at last known: approx using entry (no live price here)
            pass
        return {
            "trades": n, "wins": len(wins), "losses": len(losses),
            "winrate": round(wr, 4), "pf": round(pf, 4), "net": round(net, 2),
            "open_count": len(self.open_trades),
            "last_closed": self.closed[-5:][::-1],
        }


if __name__ == "__main__":
    pt = PaperTracker("memory/paper_test.json")
    pt.open_trades.clear(); pt.closed.clear()
    pt.open_trade(1, 2000.0, 0.5, 2.5, 0.01)
    pt.open_trade(-1, 2000.0, 0.5, 2.5, 0.01)
    pt.update(2002.6)   # buy TP hit, sell SL hit
    pt.update(2002.6)
    print(pt.stats())
