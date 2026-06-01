# FridayTrader 🤖

An autonomous AI paper trading agent built with Claude (Anthropic API) that makes real-time buy/sell decisions on NASDAQ stocks using technical analysis.

## Features
- **RSI** — avoids overbought stocks, targets oversold opportunities
- **MACD** — only buys when momentum is building
- **Volume Analysis** — confirms signals with volume vs 20-day average
- **5-Day Trend** — week-level momentum, not just daily noise
- **News Sentiment** — live headlines via Yahoo Finance RSS
- **Stop-loss / Take-profit** — automatic risk management (5% / 15%)
- **Live Dashboard** — Flask + ngrok web UI with performance vs S&P 500
- **Obsidian Logs** — every decision logged with full reasoning

## Stack
- Python 3.9
- Anthropic Claude API (claude-sonnet-4-6)
- yfinance for market data
- Flask for dashboard
- schedule for automation

## How It Works
1. Fetches 60 days of price/volume data per stock
2. Calculates RSI, MACD, volume ratio, 5-day trend
3. Sends technical context + news to Claude for decision
4. Executes paper trades and logs reasoning to Obsidian
5. Runs automatically Mon-Fri at market open and midday

## Results
- Starting capital: $10,000
- Currently tracking: AAPL, NVDA, TSLA, MSFT, GOOGL
- Live P&L tracked vs S&P 500 benchmark

*Built by Vedant Nogaja — Singapore, 2026*
