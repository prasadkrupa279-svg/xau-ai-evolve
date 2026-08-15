# 🥇 XAUUSD — AI Agent Evolutionary Backtesting + Live Trading

An evolutionary AI system for **XAUUSD (Gold)** that backtests 3 years of M1 data,
improves its strategy with a genetic algorithm (winrate × profit-factor × trades,
all at once), and runs 24/7 on Render with a live dashboard, paper/demo trading and
Telegram alerts.

> ⚖️ **Honest by design.** All numbers are real no-look-ahead backtests. 1:5 RR makes
> ~16–34% winrate mathematically honest. **75%+ winrate at 1:5 is impossible** and is
> never claimed. LIVE trading is **hard-blocked** — demo/paper only.

## 🔗 Live
- **Dashboard (24/7 evolving):** https://xau-ai-evolve.onrender.com
- **Source:** https://github.com/prasadkrupa279-svg/xau-ai-evolve
- **Render:** https://dashboard.render.com/web/srv-da058as9v7es738kcan0

---

## 📦 Deliverables

| File | Role |
|---|---|
| `indicators_lib.py` | 37 vote-sources (15 indicators × periods), closed-bar, look-ahead-safe |
| `ppa_engine.py` | AB-Touch POI zones + entry/exit backtest with **SL-guard**, 1:5 RR |
| `genome.py` | Agent genome + true dedup key + 3-objective fitness |
| `evolution.py` | Genetic engine: elitism, **adaptive mutation**, dedup, global memory, consensus |
| `ai_agent_daemon.py` | 24/7 evolution daemon (standalone or background thread) |
| `backtest_reproduce.py` | Standalone, fully reproducible single-agent backtest CLI |
| `realtime_alerter.py` | Live feed + paper execution + Telegram alerts |
| `live_tracker.py` | Paper-trade result tracker (SL-guard) |
| `demo_trader.py` | cTrader **demo** bridge (live hard-blocked, simulated fallback) |
| `dashboard.py` | Flask dashboard (evolution + UI in one process) for Render |
| `pine/consensus.pine` | TradingView Pine v5 of the 37-vote consensus |
| `Dockerfile` / `Procfile` / `requirements.txt` | Render deployment |

## 🚀 Run locally

```bash
pip install -r requirements.txt
python3 dashboard.py                 # http://localhost:5000  (evolves 24/7)
python3 backtest_reproduce.py        # standalone honest backtest
python3 ai_agent_daemon.py           # headless 24/7 daemon
```

## 📈 Add YOUR real data (zero code changes)

Drop these tab-separated files into `data/`:
```
GOLD_2023_2024.csv   GOLD_2024_2025.csv   GOLD_2025_2026.csv
# columns: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
```
`load_data()` merges, parses datetime, sorts, dedups. Until then it runs on a
clearly-labelled **synthetic** feed so the system is testable out of the box.

## 🧬 How the AI works

1. **37 votes** per bar from RSI/Stoch/WPR/MFI/CMF/Force/OBV/VWAP/Aroon/TSI/Ultimate/
   BOP/Vortex/ZLEMA/Elder (multiple periods).
2. Each agent = a genome: which votes to use (37-bit mask), thresholds, SL/TP, trend
   filter, session. **Full-genome key ⇒ true dedup** (no duplicate strategies).
3. **AB-Touch POI**: active swing-pivot / FVG zone that price wicked back into +
   trend filter + `|net votes| ≥ ind_conf`. Fill at next open (no look-ahead).
4. **SL-guard exit**: same-bar SL beats TP. TP = full 1:5 win. No trailing/partials.
5. **Fitness** = `winrate³ × PF^1.5 × trades_factor × net(mild)` — all four together.
6. **Evolution**: 50 unique agents/gen, elitism, **adaptive mutation** (rises on
   stagnation, falls while improving — never stuck), champion saved to
   `GLOBAL_AI_MEMORY` (restart-safe).
7. **Consensus**: top-K agents vote the latest closed bar → buy/sell/flat.
8. **Live**: paper trades on the feed (SL-guard), optional cTrader **demo** mirror
   (0.01 lot, SL $0.5 / TP $2.5), Telegram alerts.

## ⚙️ Env (see `.env.example`)
`DATA_DIR`, `MEMORY_PATH`, `GEN_SLEEP`, `ENABLE_PAPER`, `TRADE_MODE`(demo only),
`CTRADER_*`, `TELEGRAM_*`, `PORT`.

## ⚠️ Safety / Disclaimer
- **LIVE trading is impossible** here (`demo_trader.NEVER_LIVE` + `TRADE_MODE` gate).
- Paper/demo only. Synthetic numbers are for testing, not a performance claim.
- Not financial advice. Trading XAUUSD is high-risk; past performance ≠ future results.
- Render **free tier sleeps** after ~15 min idle (self keep-alive pings help). For
  guaranteed 24/7 use a paid instance / background worker.
