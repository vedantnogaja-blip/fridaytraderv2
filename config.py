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
