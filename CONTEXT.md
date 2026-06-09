# FridayTrader Project Context

## Current State
- Portfolio: $10,216.17 (+2.16%) from $10,000
- Cash: $12.87 (nearly fully invested)
- Holdings: AAPL (6), NVDA (14), MSFT (5), GOOGL (2), NNE (33), DCO (8)

## Watchlist (17 stocks)
Tech: AAPL, NVDA, TSLA, MSFT, GOOGL
Nuclear: NNE, LEU, CEG, CCJ
Defense: RDW, DCO
Retail: COST, TJX
Food: CAVA, CMG
AI Infra: VRT, CRDO

## Files
- friday_trader_v3.py — main trading agent
- server.py — Flask dashboard on port 9090
- scheduler.py — runs Mon-Fri 22:00 and 02:00 SGT
- dashboard.html — web UI

## Done (2026-06-02)
- server.py /prices now fetches all 17 stocks (WATCHLIST), with a 60s price cache that survives transient fetch failures
- Live Market tab fully wired: loadMarket() fetches /prices + /indices + portfolio, renders sector-grouped watchlist (renderWatchlist), index cards, and full 17-row price table with positions
- Added missing --bg2 CSS var used by watchlist cards

## Pending
- (none open)
