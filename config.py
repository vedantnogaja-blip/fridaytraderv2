# Single source of truth for the tradable universe.
# Imported by friday_trader_v3.py (the agent) and server.py (the dashboard).
WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "NNE", "LEU", "RDW",
             "DCO", "CEG", "CCJ", "COST", "TJX", "CAVA", "CMG", "VRT", "CRDO"]

# Advanced strategy layer weights (passed to combine_signals in advanced_strategies.py).
# Each weight scales that layer's [-3,+3] score before summing.
ADVANCED_LAYER_WEIGHTS = {"regime": 1.0, "pairs": 1.0, "ml": 1.0}

# Hard cap on the advanced signal's net contribution to the blended score.
# Technical score range is roughly [-8, +8]; keeping this at 2 makes advanced a nudge, not dominant.
ADVANCED_CAP = 2

# ATR-based stop and target multipliers (replace fixed % stops)
ATR_STOP_MULT   = 2.0   # stop   = entry_price - ATR_14 * ATR_STOP_MULT
ATR_TARGET_MULT = 4.0   # target = entry_price + ATR_14 * ATR_TARGET_MULT

# Drawdown circuit breaker (fraction of rolling peak portfolio value)
DRAWDOWN_CIRCUIT_BREAKER = 0.08   # halt new BUYs when drawdown from peak exceeds 8%
DRAWDOWN_RESUME_PCT      = 0.05   # resume when drawdown recovers below 5% from peak
