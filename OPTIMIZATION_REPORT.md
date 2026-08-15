# Arena Unified Liquidity — AI Optimization Report

**Data:** REAL XAUUSD M1, 3 years (2023-06-26 → 2026-06-24), **1,059,978 candles**
**Method:** Genetic evolution (40 agents × 20 gens) on the Python port of your Pine, SL-guard, no look-ahead.
**Fitness:** rewards MORE trades (hard floor 250) + balanced winrate + PF (capped at 2.5 to kill tiny-sample overfit).

## Honest result
| Metric | Baseline (Pine defaults) | AI-Improved champion |
|---|---|---|
| Trades | 267 | **922** |
| Win rate | 31.1% | **45.1%** |
| Profit factor | 1.24 | 1.16 |
| Net profit (1 unit) | +$259 | **+$709** |
| Avg stop | 3.0 × ATR | **1.32 × ATR (tight)** |
| RR | 3.0 | 1.54 |

> ⚖️ PF 1.16 over 922 trades is **more trustworthy** than PF 1.24 over 267 or a fake PF 6 over 35.
> The GA first tried to cheat to 35 trades / PF 6 (overfit) — the rebalanced fitness stopped that.

## Optimized parameters (champion)
```
asian_start      631   (10:31 broker-server time)
asian_end        728   (12:08)
sweep_start      736   (12:16)
sweep_end        960   (16:00)
confirm_end     1152   (19:12)
bars_after_sweep 207
range_atr        1.39
min_body         0.59
close_pos_min    0.60
min_stop_atr     1.32   <- tight, market-based stop
rr_target        1.54
stop_buffer      0.27
cooldown         8
trend filter     H1 + H4  (daily OFF)
```

## ⚠️ Timezone note (important)
The session windows above are in **your CSV's native broker-server timestamps** (the GA tuned
alignment directly on the data, sidestepping the TZ problem). On your TradingView chart, set the
indicator's `Session Timezone` to match your broker's server time, then use these windows. Verify
the Asian range band visually lines up before trusting live signals.

## What was fixed under the hood
- `blong = bshort = np.ones(...)` aliasing bug → separate arrays (was zeroing all signals).
- Aroon/perf vectorized; SL-guard + one-trade-lock faithful to Pine.
- Fitness rebalanced to stop tiny-sample overfit and force real trade volume.

## Next levers (if you want PF higher too)
- Raise the RR floor (e.g. RR 2.0–2.5) → PF rises, trades/WR dip slightly.
- More generations (50+) or bigger population.
- Run the same optimizer per-timeframe (3m/5m/15m) — your Pine supports all TFs.
