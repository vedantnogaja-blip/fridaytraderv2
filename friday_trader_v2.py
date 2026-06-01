import anthropic
import yfinance as yf
import json
import os
import schedule
import time
import requests
from datetime import datetime, timedelta
import pytz

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VAULT_PATH = os.path.expanduser("~/Documents/FridayTrader")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
STARTING_CASH = 10000.00
WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]
PORTFOLIO_FILE = f"{VAULT_PATH}/portfolio.json"
PERFORMANCE_FILE = f"{VAULT_PATH}/performance.json"
STOP_LOSS_PCT = 0.05       # Sell if down 5% from avg purchase price
TAKE_PROFIT_PCT = 0.15     # Sell if up 15% from avg purchase price
MAX_POSITION_PCT = 0.20    # Never more than 20% of cash in one trade
SP500_SYMBOL = "^GSPC"

# ─── PORTFOLIO ────────────────────────────────────────────────────────────────
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "holdings": {}, "trades": [], "sessions": 0}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

# ─── PERFORMANCE TRACKING ─────────────────────────────────────────────────────
def load_performance():
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, 'r') as f:
            return json.load(f)
    return {
        "snapshots": [],
        "win_trades": 0,
        "loss_trades": 0,
        "total_realized_pnl": 0,
        "sp500_start": None
    }

def save_performance(perf):
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(perf, f, indent=2)

def record_snapshot(portfolio, current_prices):
    """Record a daily portfolio value snapshot for charting"""
    perf = load_performance()
    total = portfolio["cash"]
    for sym, h in portfolio["holdings"].items():
        price = current_prices.get(sym, h["avg_price"])
        total += h["shares"] * price

    # Record SP500 starting value if not set
    if perf["sp500_start"] is None:
        try:
            sp = yf.Ticker(SP500_SYMBOL).history(period="1d")
            if not sp.empty:
                perf["sp500_start"] = float(sp["Close"].iloc[-1])
        except:
            pass

    perf["snapshots"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "value": round(total, 2),
        "cash": round(portfolio["cash"], 2),
        "pnl": round(total - STARTING_CASH, 2)
    })
    save_performance(perf)

# ─── MARKET DATA ──────────────────────────────────────────────────────────────
def is_market_open():
    ny = pytz.timezone("America/New_York")
    now = datetime.now(ny)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 570 <= minutes < 960  # 9:30am - 4:00pm

def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return None
        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        change_pct = ((current - prev) / prev) * 100

        # Get additional data for better analysis
        info = {}
        try:
            info = ticker.fast_info
        except:
            pass

        return {
            "symbol": symbol,
            "price": round(current, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(hist["Volume"].iloc[-1]),
            "avg_volume": int(hist["Volume"].mean()),
            "high": round(float(hist["High"].iloc[-1]), 2),
            "low": round(float(hist["Low"].iloc[-1]), 2),
            "week_high": round(float(hist["High"].max()), 2),
            "week_low": round(float(hist["Low"].min()), 2),
        }
    except Exception as e:
        print(f"  ⚠️  Error fetching {symbol}: {e}")
        return None

def get_sp500_change():
    try:
        sp = yf.Ticker(SP500_SYMBOL).history(period="2d")
        if len(sp) >= 2:
            current = float(sp["Close"].iloc[-1])
            prev = float(sp["Close"].iloc[-2])
            return round(((current - prev) / prev) * 100, 2)
    except:
        pass
    return None

# ─── NEWS SENTIMENT ───────────────────────────────────────────────────────────
def get_news_sentiment(symbol, client):
    """Use Claude to assess news sentiment for a stock"""
    try:
        # Get recent news headlines via yfinance
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return "No recent news available.", "NEUTRAL"

        # Take top 5 headlines
        headlines = []
        for item in news[:5]:
            title = item.get("content", {}).get("title", "") or item.get("title", "")
            if title:
                headlines.append(title)

        if not headlines:
            return "No recent headlines found.", "NEUTRAL"

        headlines_text = "\n".join([f"- {h}" for h in headlines])

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Analyze these recent news headlines for {symbol} and give a sentiment rating.

Headlines:
{headlines_text}

Respond in exactly this format:
SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
SUMMARY: [one sentence summary of the news sentiment]"""
            }]
        )

        response = message.content[0].text.strip()
        sentiment = "NEUTRAL"
        summary = "No clear sentiment signal."

        for line in response.split("\n"):
            if line.startswith("SENTIMENT:"):
                sentiment = line.split(":")[1].strip()
            elif line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()

        return summary, sentiment, headlines

    except Exception as e:
        return "Could not fetch news.", "NEUTRAL", []

# ─── STOP LOSS CHECK ──────────────────────────────────────────────────────────
def check_stop_loss_take_profit(portfolio, symbol, current_price):
    """Check if any position needs forced stop-loss or take-profit"""
    if symbol not in portfolio["holdings"]:
        return None, None

    holding = portfolio["holdings"][symbol]
    avg_price = holding["avg_price"]
    change_from_avg = (current_price - avg_price) / avg_price

    if change_from_avg <= -STOP_LOSS_PCT:
        return "STOP_LOSS", f"Stop-loss triggered: {symbol} down {abs(change_from_avg*100):.1f}% from avg purchase price of ${avg_price}"

    if change_from_avg >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT", f"Take-profit triggered: {symbol} up {change_from_avg*100:.1f}% from avg purchase price of ${avg_price}"

    return None, None

# ─── CLAUDE DECISION ──────────────────────────────────────────────────────────
def ask_claude(stock_data, portfolio, news_summary, news_sentiment, sp500_change, client):
    sp500_context = f"S&P 500 is {'up' if sp500_change and sp500_change > 0 else 'down'} {abs(sp500_change):.2f}% today." if sp500_change else ""

    # Check if we already hold this stock
    holding_info = ""
    if stock_data["symbol"] in portfolio["holdings"]:
        h = portfolio["holdings"][stock_data["symbol"]]
        pnl_pct = ((stock_data["price"] - h["avg_price"]) / h["avg_price"]) * 100
        holding_info = f"Currently holding {h['shares']} shares, avg price ${h['avg_price']}, unrealized P&L: {pnl_pct:+.1f}%"

    avg_vol = stock_data.get("avg_volume", 0)
    volume_ratio = stock_data["volume"] / avg_vol if avg_vol > 0 else 1

    prompt = f"""You are a disciplined paper trading analyst. Make a trading decision for {stock_data['symbol']}.

MARKET DATA:
- Price: ${stock_data['price']} ({stock_data['change_pct']:+.2f}% today)
- Volume: {stock_data['volume']:,} ({volume_ratio:.1f}x average volume)
- Day range: ${stock_data['low']} - ${stock_data['high']}
- 5-day range: ${stock_data['week_low']} - ${stock_data['week_high']}
{sp500_context}

NEWS SENTIMENT: {news_sentiment}
{news_summary}

PORTFOLIO:
- Available cash: ${portfolio['cash']:.2f}
- Max trade size (20% rule): ${portfolio['cash'] * MAX_POSITION_PCT:.2f}
- {holding_info if holding_info else 'No current position in this stock'}
- All holdings: {json.dumps({k: v['shares'] for k, v in portfolio['holdings'].items()})}

RULES:
1. BUY only if: price momentum is positive AND news sentiment is BULLISH or NEUTRAL AND volume is above average
2. SELL only if: you hold shares AND (news is BEARISH OR momentum is negative OR position is up >10%)
3. HOLD if signals are mixed or weak
4. Never exceed 20% of cash on one trade
5. Diversify — avoid putting too much in one sector

Respond ONLY in this exact format:
DECISION: [BUY/SELL/HOLD]
SHARES: [integer, 0 if HOLD]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [2-3 sentences explaining your decision incorporating both price action and news]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

def parse_decision(response):
    lines = response.strip().split("\n")
    decision, shares, confidence, reasoning = "HOLD", 0, "LOW", ""
    for line in lines:
        if line.startswith("DECISION:"):
            decision = line.split(":", 1)[1].strip()
        elif line.startswith("SHARES:"):
            try:
                shares = int(line.split(":", 1)[1].strip())
            except:
                shares = 0
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip()
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return decision, shares, confidence, reasoning

# ─── EXECUTE TRADE ────────────────────────────────────────────────────────────
def execute_trade(portfolio, decision, shares, symbol, price, reason=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    perf = load_performance()

    if decision in ["BUY"] and shares > 0:
        cost = shares * price
        if cost <= portfolio["cash"]:
            portfolio["cash"] = round(portfolio["cash"] - cost, 2)
            if symbol in portfolio["holdings"]:
                # Average down/up
                existing = portfolio["holdings"][symbol]
                total_shares = existing["shares"] + shares
                avg = ((existing["shares"] * existing["avg_price"]) + (shares * price)) / total_shares
                portfolio["holdings"][symbol] = {"shares": total_shares, "avg_price": round(avg, 2)}
            else:
                portfolio["holdings"][symbol] = {"shares": shares, "avg_price": round(price, 2)}
            trade = {"action": "BUY", "symbol": symbol, "shares": shares, "price": price, "time": timestamp, "reason": reason}
            portfolio["trades"].append(trade)
            print(f"  ✅ BUY {shares} shares of {symbol} @ ${price} (cost: ${cost:.2f})")
        else:
            print(f"  ❌ Insufficient cash: need ${cost:.2f}, have ${portfolio['cash']:.2f}")

    elif decision in ["SELL", "STOP_LOSS", "TAKE_PROFIT"] and shares > 0:
        if symbol in portfolio["holdings"]:
            available = portfolio["holdings"][symbol]["shares"]
            sell_shares = min(shares, available)
            proceeds = sell_shares * price
            avg_price = portfolio["holdings"][symbol]["avg_price"]
            realized_pnl = (price - avg_price) * sell_shares

            portfolio["cash"] = round(portfolio["cash"] + proceeds, 2)
            portfolio["holdings"][symbol]["shares"] -= sell_shares
            if portfolio["holdings"][symbol]["shares"] <= 0:
                del portfolio["holdings"][symbol]

            # Track win/loss
            if realized_pnl > 0:
                perf["win_trades"] += 1
            else:
                perf["loss_trades"] += 1
            perf["total_realized_pnl"] = round(perf.get("total_realized_pnl", 0) + realized_pnl, 2)
            save_performance(perf)

            trade = {"action": "SELL", "symbol": symbol, "shares": sell_shares, "price": price,
                     "time": timestamp, "reason": reason, "realized_pnl": round(realized_pnl, 2)}
            portfolio["trades"].append(trade)
            emoji = "🛑" if decision == "STOP_LOSS" else "🎯" if decision == "TAKE_PROFIT" else "✅"
            print(f"  {emoji} SELL {sell_shares} shares of {symbol} @ ${price} | P&L: ${realized_pnl:+.2f}")
        else:
            print(f"  ❌ No position in {symbol} to sell")

# ─── OBSIDIAN LOGGING ─────────────────────────────────────────────────────────
def log_trade_to_obsidian(portfolio, symbol, decision, shares, price, reasoning,
                           confidence, stock_data, news_summary, news_sentiment, headlines):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    filename = f"{VAULT_PATH}/Trades/{date_str}-{symbol}.md"

    headlines_md = "\n".join([f"- {h}" for h in (headlines or [])]) or "- No headlines found"

    content = f"""# {symbol} — {decision} — {date_str}

## Market Data
| Metric | Value |
|--------|-------|
| Price | ${price} |
| Change | {stock_data['change_pct']:+.2f}% |
| Volume | {stock_data['volume']:,} |
| Day Range | ${stock_data['low']} – ${stock_data['high']} |
| Time | {time_str} |

## News Sentiment
**Rating:** {news_sentiment}
**Summary:** {news_summary}

### Headlines
{headlines_md}

## AI Decision
| Field | Value |
|-------|-------|
| Action | **{decision}** |
| Shares | {shares} |
| Confidence | {confidence} |

## Claude's Reasoning
{reasoning}

## Portfolio After Trade
- **Cash:** ${round(portfolio['cash'], 2):,}
- **Holdings:** {json.dumps({k: f"{v['shares']} shares @ ${v['avg_price']}" for k, v in portfolio['holdings'].items()}, indent=2)}
"""
    os.makedirs(f"{VAULT_PATH}/Trades", exist_ok=True)
    with open(filename, "w") as f:
        f.write(content)
    print(f"  📝 Logged to Obsidian")

def write_daily_report(portfolio, current_prices, sp500_change):
    """Write a daily performance report to Obsidian"""
    perf = load_performance()
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{VAULT_PATH}/Reports/{date_str}-report.md"

    total = portfolio["cash"]
    holdings_detail = []
    for sym, h in portfolio["holdings"].items():
        price = current_prices.get(sym, h["avg_price"])
        value = h["shares"] * price
        pnl = (price - h["avg_price"]) / h["avg_price"] * 100
        total += value
        holdings_detail.append(f"| {sym} | {h['shares']} | ${h['avg_price']} | ${price} | {pnl:+.1f}% | ${value:.2f} |")

    pnl = total - STARTING_CASH
    pnl_pct = pnl / STARTING_CASH * 100
    win_rate = 0
    total_closed = perf.get("win_trades", 0) + perf.get("loss_trades", 0)
    if total_closed > 0:
        win_rate = (perf["win_trades"] / total_closed) * 100

    sp500_line = f"S&P 500 today: {sp500_change:+.2f}%" if sp500_change else ""

    holdings_table = "\n".join(holdings_detail) if holdings_detail else "| — | — | — | — | — | — |"

    content = f"""# Daily Report — {date_str}

## Portfolio Summary
| Metric | Value |
|--------|-------|
| Total Value | ${total:,.2f} |
| Cash | ${portfolio['cash']:,.2f} |
| Starting Capital | ${STARTING_CASH:,.2f} |
| Total P&L | ${pnl:+,.2f} ({pnl_pct:+.2f}%) |
| Sessions Run | {portfolio.get('sessions', 0)} |

## Performance Stats
| Metric | Value |
|--------|-------|
| Win Rate | {win_rate:.1f}% |
| Winning Trades | {perf.get('win_trades', 0)} |
| Losing Trades | {perf.get('loss_trades', 0)} |
| Total Realized P&L | ${perf.get('total_realized_pnl', 0):+,.2f} |
{sp500_line}

## Current Holdings
| Symbol | Shares | Avg Price | Current | P&L % | Value |
|--------|--------|-----------|---------|-------|-------|
{holdings_table}

## Recent Trades
{chr(10).join([f"- {t['time']}: **{t['action']}** {t.get('shares',0)} {t['symbol']} @ ${t['price']}" for t in portfolio['trades'][-10:]])}
"""
    os.makedirs(f"{VAULT_PATH}/Reports", exist_ok=True)
    with open(filename, "w") as f:
        f.write(content)
    print(f"\n📊 Daily report written to Obsidian")

# ─── BACKTESTING ──────────────────────────────────────────────────────────────
def run_backtest(symbol="AAPL", period="1y"):
    """
    Backtest a simple momentum strategy on historical data.
    Buys when price is up >0.5% with above-average volume.
    Sells when price drops >5% from buy price (stop-loss) or up >15% (take-profit).
    """
    print(f"\n{'='*55}")
    print(f"📈 BACKTESTING: {symbol} — {period}")
    print(f"{'='*55}")

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
    except Exception as e:
        print(f"Error fetching backtest data: {e}")
        return

    if len(hist) < 10:
        print("Not enough historical data.")
        return

    cash = STARTING_CASH
    shares = 0
    buy_price = 0
    trades = []
    avg_volume = hist["Volume"].mean()

    for i in range(1, len(hist)):
        today = hist.iloc[i]
        yesterday = hist.iloc[i-1]
        price = float(today["Close"])
        prev_price = float(yesterday["Close"])
        volume = float(today["Volume"])
        change_pct = ((price - prev_price) / prev_price) * 100
        date = hist.index[i].strftime("%Y-%m-%d")

        # Check stop-loss / take-profit if holding
        if shares > 0:
            pnl_pct = (price - buy_price) / buy_price
            if pnl_pct <= -STOP_LOSS_PCT:
                proceeds = shares * price
                realized = (price - buy_price) * shares
                cash += proceeds
                trades.append({"date": date, "action": "STOP_LOSS", "price": price,
                               "shares": shares, "pnl": round(realized, 2)})
                shares = 0
                buy_price = 0
                continue
            if pnl_pct >= TAKE_PROFIT_PCT:
                proceeds = shares * price
                realized = (price - buy_price) * shares
                cash += proceeds
                trades.append({"date": date, "action": "TAKE_PROFIT", "price": price,
                               "shares": shares, "pnl": round(realized, 2)})
                shares = 0
                buy_price = 0
                continue

        # Buy signal: positive momentum + above average volume + no current position
        if shares == 0 and change_pct > 0.5 and volume > avg_volume:
            max_spend = cash * MAX_POSITION_PCT
            buy_shares = int(max_spend / price)
            if buy_shares > 0:
                cost = buy_shares * price
                cash -= cost
                shares = buy_shares
                buy_price = price
                trades.append({"date": date, "action": "BUY", "price": price,
                               "shares": buy_shares, "pnl": 0})

    # Close any open position at end
    if shares > 0:
        price = float(hist["Close"].iloc[-1])
        realized = (price - buy_price) * shares
        cash += shares * price
        trades.append({"date": hist.index[-1].strftime("%Y-%m-%d"), "action": "SELL",
                       "price": price, "shares": shares, "pnl": round(realized, 2)})

    final_value = cash
    pnl = final_value - STARTING_CASH
    pnl_pct = (pnl / STARTING_CASH) * 100
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades if t.get("pnl", 0) < 0)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    print(f"  Period:        {hist.index[0].strftime('%Y-%m-%d')} → {hist.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Starting:      ${STARTING_CASH:,.2f}")
    print(f"  Final value:   ${final_value:,.2f}")
    print(f"  Total P&L:     ${pnl:+,.2f} ({pnl_pct:+.1f}%)")
    print(f"  Total trades:  {len(trades)}")
    print(f"  Win rate:      {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Stop-losses:   {sum(1 for t in trades if t['action'] == 'STOP_LOSS')}")
    print(f"  Take-profits:  {sum(1 for t in trades if t['action'] == 'TAKE_PROFIT')}")

    # Save backtest report to Obsidian
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{VAULT_PATH}/Strategy/backtest-{symbol}-{date_str}.md"
    trades_md = "\n".join([f"| {t['date']} | {t['action']} | ${t['price']:.2f} | {t['shares']} | ${t.get('pnl',0):+.2f} |"
                           for t in trades[-20:]])

    content = f"""# Backtest: {symbol} — {period} — {date_str}

## Results
| Metric | Value |
|--------|-------|
| Period | {hist.index[0].strftime('%Y-%m-%d')} → {hist.index[-1].strftime('%Y-%m-%d')} |
| Starting Capital | ${STARTING_CASH:,.2f} |
| Final Value | ${final_value:,.2f} |
| Total P&L | ${pnl:+,.2f} ({pnl_pct:+.1f}%) |
| Total Trades | {len(trades)} |
| Win Rate | {win_rate:.1f}% |
| Wins | {wins} |
| Losses | {losses} |
| Stop-losses triggered | {sum(1 for t in trades if t['action'] == 'STOP_LOSS')} |
| Take-profits triggered | {sum(1 for t in trades if t['action'] == 'TAKE_PROFIT')} |

## Strategy Rules
- Buy when: daily change > +0.5% AND volume > average volume
- Stop-loss: sell if down {STOP_LOSS_PCT*100:.0f}% from purchase price
- Take-profit: sell if up {TAKE_PROFIT_PCT*100:.0f}% from purchase price
- Max position: {MAX_POSITION_PCT*100:.0f}% of available cash

## Last 20 Trades
| Date | Action | Price | Shares | P&L |
|------|--------|-------|--------|-----|
{trades_md}
"""
    os.makedirs(f"{VAULT_PATH}/Strategy", exist_ok=True)
    with open(filename, "w") as f:
        f.write(content)
    print(f"  📝 Backtest report saved to Obsidian")
    print(f"{'='*55}\n")
    return {"final_value": final_value, "pnl": pnl, "pnl_pct": pnl_pct, "win_rate": win_rate}

# ─── MAIN TRADING SESSION ─────────────────────────────────────────────────────
def run_trading_session():
    client = anthropic.Anthropic(api_key=API_KEY)
    portfolio = load_portfolio()
    portfolio["sessions"] = portfolio.get("sessions", 0) + 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    market_open = is_market_open()
    sp500_change = get_sp500_change()

    print(f"\n{'='*55}")
    print(f"🤖 FridayTrader v2 — {now}")
    print(f"📊 Market: {'🟢 OPEN' if market_open else '🔴 CLOSED'}")
    print(f"💰 Cash: ${portfolio['cash']:,.2f} | Holdings: {list(portfolio['holdings'].keys())}")
    if sp500_change:
        print(f"📈 S&P 500: {sp500_change:+.2f}% today")
    print(f"{'='*55}\n")

    current_prices = {}

    for symbol in WATCHLIST:
        print(f"📊 Analyzing {symbol}...")

        # Get stock data
        stock_data = get_stock_data(symbol)
        if not stock_data:
            print(f"  ⚠️  Could not fetch data for {symbol}, skipping\n")
            continue

        current_prices[symbol] = stock_data["price"]
        print(f"  Price: ${stock_data['price']} ({stock_data['change_pct']:+.2f}%) | Vol: {stock_data['volume']:,}")

        # ── STOP LOSS / TAKE PROFIT CHECK (runs even when market closed) ──
        trigger, trigger_msg = check_stop_loss_take_profit(portfolio, symbol, stock_data["price"])
        if trigger:
            print(f"  ⚡ {trigger_msg}")
            shares_to_sell = portfolio["holdings"][symbol]["shares"]
            execute_trade(portfolio, trigger, shares_to_sell, symbol, stock_data["price"], trigger_msg)
            log_trade_to_obsidian(portfolio, symbol, trigger, shares_to_sell,
                                   stock_data["price"], trigger_msg, "AUTO", stock_data, trigger_msg, "AUTO", [])
            continue

        # ── NEWS SENTIMENT ──
        print(f"  📰 Fetching news sentiment...")
        news_summary, news_sentiment, headlines = get_news_sentiment(symbol, client)
        print(f"  News: {news_sentiment} — {news_summary[:60]}...")

        # ── CLAUDE DECISION ──
        print(f"  🧠 Asking Claude...")
        response = ask_claude(stock_data, portfolio, news_summary, news_sentiment, sp500_change, client)
        decision, shares, confidence, reasoning = parse_decision(response)
        print(f"  Decision: {decision} {shares} shares (Confidence: {confidence})")
        print(f"  Reasoning: {reasoning[:80]}...")

        # ── EXECUTE ──
        execute_trade(portfolio, decision, shares, symbol, stock_data["price"], reasoning)

        # ── LOG TO OBSIDIAN ──
        log_trade_to_obsidian(portfolio, symbol, decision, shares, stock_data["price"],
                               reasoning, confidence, stock_data, news_summary, news_sentiment, headlines)
        print()

    # Record snapshot and save
    record_snapshot(portfolio, current_prices)
    save_portfolio(portfolio)

    # Calculate totals
    total = portfolio["cash"]
    for sym, h in portfolio["holdings"].items():
        price = current_prices.get(sym, h["avg_price"])
        total += h["shares"] * price

    pnl = total - STARTING_CASH
    perf = load_performance()
    win_rate = 0
    total_closed = perf.get("win_trades", 0) + perf.get("loss_trades", 0)
    if total_closed > 0:
        win_rate = perf["win_trades"] / total_closed * 100

    print(f"\n{'='*55}")
    print(f"📈 Session Complete!")
    print(f"💰 Cash: ${portfolio['cash']:,.2f}")
    print(f"📊 Portfolio Value: ${total:,.2f}")
    print(f"📉 Total P&L: ${pnl:+,.2f} ({pnl/STARTING_CASH*100:+.2f}%)")
    print(f"🏆 Win Rate: {win_rate:.1f}% ({perf.get('win_trades',0)}W / {perf.get('loss_trades',0)}L)")
    print(f"{'='*55}\n")

    # Write daily report
    write_daily_report(portfolio, current_prices, sp500_change)

# ─── SCHEDULER ────────────────────────────────────────────────────────────────
def run_scheduler():
    print("🚀 FridayTrader v2 Scheduler Started")
    print("📅 Runs Mon-Fri at 10:00 AM and 2:00 PM Singapore time")
    print("🛑 Stop-loss checks run every session regardless of market hours")
    print("Press Ctrl+C to stop\n")

    # Run once immediately
    run_trading_session()

    # Schedule twice daily Mon-Fri (10am and 2pm SGT)
    for day in ["monday","tuesday","wednesday","thursday","friday"]:
        getattr(schedule.every(), day).at("10:00").do(run_trading_session)
        getattr(schedule.every(), day).at("14:00").do(run_trading_session)

    while True:
        schedule.run_pending()
        time.sleep(60)

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "backtest":
            # Run backtest on all watchlist symbols
            print("Running backtests on all watchlist symbols...\n")
            for sym in WATCHLIST:
                run_backtest(sym, "1y")
        elif cmd == "once":
            run_trading_session()
        elif cmd == "schedule":
            run_scheduler()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python3 friday_trader_v2.py [once|schedule|backtest]")
    else:
        run_scheduler()
