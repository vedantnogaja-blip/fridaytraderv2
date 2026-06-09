# FridayTrader v3

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-purple?logo=anthropic)
![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo_Finance-purple)
![Flask](https://img.shields.io/badge/Dashboard-Flask-orange?logo=flask)
![Paper Trading](https://img.shields.io/badge/Mode-Paper_Trading-yellow)

An autonomous AI trading agent that runs twice daily on weekdays, combining technical signals, regime detection, statistical arbitrage, and ML to decide BUY/SELL/HOLD on a 17-stock watchlist. Every decision is logged to an Obsidian vault for review.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FridayTrader v3                       │
│                                                         │
│  config.py           ── Single source of truth          │
│  friday_trader_v3.py ── Main agent + scheduler          │
│  advanced_strategies.py ── 3-layer advanced signals     │
│  backtest.py         ── Walk-forward validation         │
│  server.py           ── Flask dashboard API             │
│  dashboard.html      ── Web UI (7 tabs)                 │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────┐
  │            Per-session flow                     │
  │                                                 │
  │  1. Download 2y price history (yfinance)        │
  │  2. Build advanced signal cache (once/day)      │
  │     ├── Regime detection (Efficiency Ratio)     │
  │     ├── Cointegrated pairs (statsmodels coint)  │
  │     └── GBM classifier (scikit-learn, 120 est) │
  │                                                 │
  │  3. Per symbol (17 stocks):                     │
  │     ├── Fetch 60d OHLCV with retry backoff      │
  │     ├── Compute RSI-14, MACD, ATR-14            │
  │     ├── Evaluate technical score [-8, +8]       │
  │     ├── Blend advanced contribution (cap ±2)    │
  │     ├── Fetch headlines (Yahoo RSS)             │
  │     ├── Ask Claude: BUY / SELL / HOLD           │
  │     ├── Execute trade + update portfolio.json   │
  │     └── Log decision to Obsidian (.md file)     │
  │                                                 │
  │  4. Update peak value, check drawdown breaker   │
  │  5. Save signals.json for dashboard debug tab   │
  └─────────────────────────────────────────────────┘
```

---

## Watchlist

| Sector | Symbols |
|---|---|
| Tech | AAPL, NVDA, TSLA, MSFT, GOOGL |
| Nuclear | NNE, LEU, CEG, CCJ |
| Defense | RDW, DCO |
| Retail | COST, TJX |
| Food | CAVA, CMG |
| AI Infra | VRT, CRDO |

---

## Strategy

### Technical signals — OOS diagnostic results

Tested over 150 OOS days, 2465 pooled symbol-observations:

| Signal | Fires | Hit rate | vs Baseline | Verdict |
|---|---|---|---|---|
| RSI < 35 (oversold) | 16% | **59.5%** | **+8.2pp** | HAS EDGE ✓ |
| RSI > 70 (overbought) | 15% | 50.8% | −0.4pp | No edge |
| MACD hist > 0 | 50% | 52.1% | +0.8pp | **No edge** ⚠ |
| 5d trend > +2% | 37% | 51.4% | +0.1pp | **No edge** ⚠ |
| Volume ≥ 1.5× avg | 9% | 51.1% | −0.1pp | No edge |
| Unconditional baseline | — | 51.3% | — | — |

**Key finding**: RSI < 35 is the only signal with statistically validated predictive power (+8.2pp, +1.39% avg 5-day return vs +0.68% baseline). MACD and 5-day trend showed zero edge. These signals are negatively correlated with RSI (r = −0.39, −0.24), so they mostly fire when RSI is not oversold. The minimum BUY score of 3 requires MACD + trend to agree — both no-edge signals — which explains weak backtest performance. When 3 bull signals agree the hit rate drops to exactly 50% (unconditional).

### Advanced signal layer (nudge, capped ±2)

| Layer | Method | OOS layer ablation |
|---|---|---|
| Regime | Efficiency Ratio → momentum or mean-reversion | Sharpe −2.62 vs tech-only −2.24 (worse) |
| Pairs | `statsmodels coint()` + rolling z-score | Sharpe −2.25 (neutral) |
| ML | `GradientBoostingClassifier`, CV AUC 0.487 | Sharpe −2.28 (neutral) |

All three layers are net noise vs technical-only in current OOS testing. They remain as optional ±2-capped nudges.

### Risk management

| Rule | Value |
|---|---|
| Stop loss | `entry − ATR₁₄ × 2.0` |
| Take profit | `entry + ATR₁₄ × 4.0` |
| Max position size | 20% of portfolio |
| Max open positions | 5 |
| Cash reserve | Maintained by 20% max position rule |
| Drawdown circuit breaker | Halt new BUYs at >8% drawdown from peak; resume at <5% |

---

## Honest backtest results

Walk-forward validation: rolling 6-month train → 1-month OOS, ~17 windows.

| Metric | FridayTrader | SPY B&H | 60/40 |
|---|---|---|---|
| OOS Return | −7.7% | +9.0% | — |
| OOS Sharpe | −2.62 | +1.17 | — |
| Max Drawdown | 8.1% | — | — |

**The strategy does not currently beat SPY or a random baseline.** The root cause is using MACD and 5-day trend as primary entry conditions — signals with zero OOS predictive power. RSI < 35 is the only validated signal, firing only 16% of the time.

Run `python3 backtest.py` for the latest numbers (saved to `backtest_results/`).

---

## Setup

### Prerequisites

```bash
pip install anthropic yfinance pandas numpy scikit-learn statsmodels flask pytz schedule python-dotenv
pip install quantstats  # optional, for HTML tearsheet
```

### Environment

Create `.env` in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
# One-shot session
python3 friday_trader_v3.py once

# Scheduled (Mon–Fri 22:00 + 02:00 SGT)
python3 friday_trader_v3.py schedule

# Dashboard → http://localhost:9090
python3 server.py

# Full walk-forward backtest (~10 min including Monte Carlo)
python3 backtest.py

# Fast backtest (skips 100-run Monte Carlo)
python3 backtest.py --fast
```

---

## Dashboard tabs

| Tab | Description |
|---|---|
| Dashboard | Portfolio value, positions heatmap, trade history, Claude's reasoning |
| Performance | Equity curve vs S&P 500, win/loss, realized P&L |
| Live Market | Real-time prices, sector watchlist, price table |
| Signal Debug | Per-symbol scores; RSI column highlighted green (validated edge), MACD/trend flagged ⚠ |
| Backtest | Walk-forward equity curve (Chart.js), per-window table, benchmark comparison |
| Risk | Circuit breaker status, ATR stops per position, drawdown chart, allocation pie |
| Skills | Architecture explainer |

---

## File layout

```
FridayTrader/
├── friday_trader_v3.py      # Main agent + scheduler
├── advanced_strategies.py   # Regime, pairs, ML layers
├── backtest.py              # Walk-forward backtester
├── config.py                # Watchlist + all hyperparameters
├── server.py                # Flask API (dashboard backend)
├── dashboard.html           # Web dashboard
├── Trades/                  # Obsidian trade logs
│   └── YYYY-MM-DD-SYM-v3.md
├── backtest_results/        # Generated — gitignored
│   ├── report_YYYYMMDD_HHMMSS.txt
│   └── backtest_latest.json
├── portfolio.json           # Live state — gitignored
├── performance.json         # Session snapshots — gitignored
└── signals.json             # Latest signal snapshot — gitignored
```

---

## API endpoints

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /portfolio.json` | Holdings, cash, trade history, circuit breaker state |
| `GET /performance.json` | Per-session snapshots + realized P&L |
| `GET /prices` | Live prices (background-polled, never blocks request) |
| `GET /signals` | Latest per-symbol: RSI, MACD, regime, ML prob, blended score |
| `GET /backtest-report` | Latest walk-forward JSON from `backtest_results/` |
| `GET /health` | last_run, circuit_breaker_active, portfolio_value, sessions_run |
| `GET /indices` | S&P 500, NASDAQ, DJIA |
