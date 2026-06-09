# Single source of truth for the tradable universe.
# Imported by friday_trader_v3.py (the agent), backtest.py, and server.py.
WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "NNE", "LEU", "RDW",
             "DCO", "CEG", "CCJ", "COST", "TJX", "CAVA", "CMG", "VRT", "CRDO"]

# Advanced strategy layer weights (passed to combine_signals in advanced_strategies.py).
ADVANCED_LAYER_WEIGHTS = {"regime": 1.0, "pairs": 1.0, "ml": 1.0}
ADVANCED_CAP = 2

# ATR-based stop and target multipliers (replace fixed % stops)
ATR_STOP_MULT   = 2.0   # stop   = entry_price - ATR_14 * ATR_STOP_MULT
ATR_TARGET_MULT = 4.0   # target = entry_price + ATR_14 * ATR_TARGET_MULT

# Drawdown circuit breaker (fraction of rolling peak portfolio value)
DRAWDOWN_CIRCUIT_BREAKER = 0.08   # halt new BUYs when drawdown from peak exceeds 8%
DRAWDOWN_RESUME_PCT      = 0.05   # resume when drawdown recovers below 5% from peak

# SPY 200-day regime filter — block ALL new BUYs when SPY < 200MA (bear market)
SPY_REGIME_WINDOW = 200

# Earnings blackout
EARNINGS_BLACKOUT_DAYS = 3   # skip symbol entirely if earnings within N calendar days
EARNINGS_EXIT_DAYS     = 2   # force SELL if holding and earnings within N calendar days

# Multi-timeframe confirmation — BUY only when weekly signal agrees
WEEKLY_RSI_PERIOD = 5     # period for weekly RSI (5-week)
WEEKLY_RSI_MAX    = 60    # weekly RSI must be < this (not overbought on weekly)
# Weekly MACD histogram must also be > 0 (checked in code)

# RSI thresholds
RSI_OVERSOLD      = 35   # primary gate: no BUY unless RSI < 35
RSI_DEEP_OVERSOLD = 25   # extra score if deeply oversold
RSI_OVERBOUGHT    = 70

# Signal thresholds for new indicators
VOLUME_SURGE_MIN  = 1.5   # volume surge multiplier vs 20-day avg
RS3M_BULL_THRESH  =  5.0  # % outperformance vs SPY over 3 months → bullish
RS3M_BEAR_THRESH  = -5.0  # % underperformance → bearish

# Buy/sell score gates (used in both live trader and backtest)
# RSI<35(+2) alone meets this threshold; volume surge(+2) makes it +4 (high confidence)
MIN_BUY_SCORE  = 2
MAX_SELL_SCORE = -3
