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

### Technical signals — v2 winning stack

Full signal ablation across all configurations (OOS 2025-10-31 → 2026-06-08):

| Config | OOS Sharpe | OOS Return | Trades | Verdict |
|---|---|---|---|---|
| v1: MACD + 5d trend | −1.11 | −9.7% | 9 | **Dropped** — zero OOS edge |
| RSI gate only | −1.38 | −13.6% | 5 | Not enough trades |
| **RSI gate + volume surge** | **−0.19** | **−4.6%** | **18** | **Best individual config** |
| RSI gate + 50MA | −1.74 | −11.2% | 4 | 50MA blocks oversold trades |
| RSI gate + all three | −1.99 | −10.7% | 3 | Over-filtered |
| SPY B&H | +1.17 | +9.0% | — | Benchmark |

**Walk-forward aggregate (17 windows, 2024–2026):**

| Metric | v1 (MACD+trend) | **v2 (RSI+volume)** | SPY B&H | 60/40 |
|---|---|---|---|---|
| OOS Return | −2.4% | **+4.8%** | +25.0% | +17.3% |
| OOS Sharpe | 0.09 | **0.30** | 0.96 | 1.06 |
| Positive windows | 9/17 | **11/17** | — | — |
| Max Drawdown | 25.8% | 25.3% | 18.8% | 11.3% |

**Signal design decisions:**
- **MACD**: dropped — 0.8pp OOS hit-rate edge, confirmed zero predictive value
- **5-day trend**: dropped — 0.1pp edge, confirmed zero predictive value
- **50MA**: informational only, not scored — blocks RSI<35 trades (stock is below MA by definition when oversold, so 50MA < price gives a −1 penalty that kills valid entries)
- **3m RS vs SPY**: informational only, not scored — same blocking problem
- **Volume surge ≥1.5×**: scored (+2) — confirmed confirming signal for oversold entries
- **RSI < 35**: primary hard gate AND scored (+2/+3) — only validated signal (+8.2pp hit rate)

Note: MIN_BUY_SCORE=2 means RSI<35 alone qualifies for a BUY. Volume surge (+2) raises the score to +4, giving priority in slot allocation. The previous ablation that showed Sharpe +1.68 used MIN_BUY_SCORE=4 (required both signals simultaneously). The walk-forward Sharpe improvement (0.09→0.30) is robust across 2 years and 17 windows.

### Additional risk filters (v2)

| Filter | Behavior |
|---|---|
| SPY 200-day regime | Block ALL new BUYs when SPY < 200MA (bear market) |
| Earnings blackout | Skip symbol within 3 days of earnings; force SELL within 2 days |
| Weekly MTF | BUY requires weekly RSI(5) < 60 AND weekly MACD hist > 0 |

### Risk management

| Rule | Value |
|---|---|
| Stop loss | `entry − ATR₁₄ × 2.0` |
| Take profit | `entry + ATR₁₄ × 4.0` |
| Max position size | 20% of portfolio |
| Max open positions | 5 |
| Drawdown circuit breaker | Halt BUYs at >8% drawdown from peak; resume at <5% |

---

## Honest backtest results

Walk-forward validation: rolling 6-month train → 1-month OOS, 17 windows over 2 years.

| Metric | v2 FridayTrader | SPY B&H | 60/40 |
|---|---|---|---|
| OOS Return | +4.8% | +25.0% | +17.3% |
| OOS Sharpe | 0.30 | 0.96 | 1.06 |
| Max Drawdown | 25.3% | 18.8% | 11.3% |
| Win Rate | 38.9% | — | — |
| Positive windows | 11/17 | — | — |

**The strategy outperforms its prior version (v1 Sharpe 0.09 → v2 Sharpe 0.30) but still lags SPY.** The RSI+volume stack is the best found so far. Overfitting risk is real — 500 trading days of data across 17 symbols is limited for robust signal validation.

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
