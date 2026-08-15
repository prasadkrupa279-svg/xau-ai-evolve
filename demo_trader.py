"""
demo_trader.py
==============
SAFE cTrader demo-order bridge.

  - TRADE_MODE must be 'demo'. Anything else (incl. 'live') is REFUSED.
    LIVE trading is hard-blocked by design (per spec safe-guard).
  - Needs cTrader OpenAPI creds (env). If absent -> SIMULATED demo fills so the
    system still runs end-to-end; nothing real is ever sent.
  - The real cTrader OpenAPI uses protobuf over TLS/WebSocket; the integration
    point is clearly marked below. Fill `_ctrader_send_demo_order` to go live
    against the DEMO host only.

NEVER wire this to a live account.
"""
from __future__ import annotations
import os, time, json, uuid

DEMO_HOST = "demo.ctraderapi.com:5035"   # cTrader OpenAPI demo host
NEVER_LIVE = True                         # hard safety flag


class DemoTrader:
    def __init__(self):
        self.mode = os.environ.get("TRADE_MODE", "off").lower()
        self.client_id = os.environ.get("CTRADER_CLIENT_ID", "")
        self.secret = os.environ.get("CTRADER_CLIENT_SECRET", "")
        self.token = os.environ.get("CTRADER_ACCESS_TOKEN", "")
        self.account = os.environ.get("CTRADER_ACCOUNT_ID", "")
        self.connected = False
        self.simulated_fills: list[dict] = []

    # ---------- safety gate ----------
    def _safety(self):
        if NEVER_LIVE and self.mode == "live":
            raise RuntimeError("LIVE trading is hard-blocked by design. Set TRADE_MODE=demo.")
        if self.mode != "demo":
            raise RuntimeError(f"TRADE_MODE={self.mode!r} not allowed. Use 'demo' only.")

    def has_ctrader_creds(self) -> bool:
        return bool(self.client_id and self.secret and self.token and self.account)

    # ---------- public ----------
    def place_market_order(self, side: int, lot: float, sl_price: float, tp_price: float,
                           symbol: str = "XAUUSD", price_ref: float = 0.0) -> dict:
        self._safety()
        if lot > 0.05:
            # spec: demo uses 0.01 lot; keep it tiny & safe
            lot = 0.01
        side_str = "BUY" if side > 0 else "SELL"

        if self.has_ctrader_creds():
            # ---- REAL cTrader DEMO integration point (demo host only) ----
            # Connect via OpenAPI protobuf, app auth, send NewOrderRequest to
            # the DEMO account. Implemented as a guarded stub:
            try:
                return self._ctrader_send_demo_order(symbol, side_str, lot, sl_price, tp_price)
            except Exception as e:
                # NEVER escalate to live; fall back to simulated
                print(f"[demo_trader] cTrader demo call failed ({e}); simulated fallback.")
                return self._simulate(symbol, side_str, lot, sl_price, tp_price, price_ref, note="fallback")
        # ---- simulated (no creds) ----
        return self._simulate(symbol, side_str, lot, sl_price, tp_price, price_ref)

    # ---------- internals ----------
    def _ctrader_send_demo_order(self, symbol, side, lot, sl, tp):  # pragma: no cover
        """TODO: implement cTrader OpenAPI (proto) NewOrder on DEMO host.
        Until implemented this raises so we fall back to simulated safely."""
        raise NotImplementedError("cTrader OpenAPI client not wired (demo host).")

    def _simulate(self, symbol, side, lot, sl, tp, price_ref, note="simulated"):
        rec = {
            "id": uuid.uuid4().hex[:8], "mode": "demo", "venue": "simulated",
            "symbol": symbol, "side": side, "lot": lot, "sl": sl, "tp": tp,
            "price_ref": price_ref, "note": note,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.simulated_fills.append(rec)
        return rec


if __name__ == "__main__":
    os.environ["TRADE_MODE"] = "demo"
    dt = DemoTrader()
    print("creds?", dt.has_ctrader_creds())
    print(dt.place_market_order(1, 0.01, 1999.5, 2002.5, price_ref=2000.0))
    # live must be blocked
    os.environ["TRADE_MODE"] = "live"
    dt2 = DemoTrader()
    try:
        dt2.place_market_order(1, 0.01, 1, 2)
        print("ERROR: live was NOT blocked!")
    except RuntimeError as e:
        print("LIVE blocked OK:", e)
