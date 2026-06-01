import anthropic
import yfinance as yf
import json
import os
from datetime import datetime

VAULT_PATH = os.path.expanduser("~/Documents/FridayTrader")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
STARTING_CASH = 10000.00
WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]
PORTFOLIO_FILE = f"{VAULT_PATH}/portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "holdings": {}, "trades": []}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5d")
        if hist.empty:
            return None
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(hist['Volume'].iloc[-1])
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def ask_claude(stock_data, portfolio):
    client = anthropic.Anthropic(api_key=API_KEY)
    prompt = f"""You are a paper trading assistant. Analyze this stock and decide whether to BUY, SELL, or HOLD.

Stock Data:
- Symbol: {stock_data['symbol']}
- Current Price: ${stock_data['price']}
- Change today: {stock_data['change_pct']}%
- Volume: {stock_data['volume']}

Portfolio:
- Available Cash: ${portfolio['cash']}
- Current Holdings: {json.dumps(portfolio['holdings'])}

Rules:
- Never spend more than 20% of available cash on one trade
- Only buy if change is showing momentum
- Sell if a holding is down more than 5% from purchase price
- Always explain your reasoning in 2-3 sentences

Respond in this exact format:
DECISION: [BUY/SELL/HOLD]
SHARES: [number or 0]
REASONING: [your explanation]"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def parse_decision(response):
    lines = response.strip().split('\n')
    decision = "HOLD"
    shares = 0
    reasoning = ""
    for line in lines:
        if line.startswith("DECISION:"):
            decision = line.split(":")[1].strip()
        elif line.startswith("SHARES:"):
            try:
                shares = int(line.split(":")[1].strip())
            except:
                shares = 0
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return decision, shares, reasoning

def execute_trade(portfolio, decision, shares, symbol, price):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trade = None

    if decision == "BUY" and shares > 0:
        cost = shares * price
        if cost <= portfolio["cash"]:
            portfolio["cash"] -= cost
            portfolio["cash"] = round(portfolio["cash"], 2)
            if symbol in portfolio["holdings"]:
                portfolio["holdings"][symbol]["shares"] += shares
            else:
                portfolio["holdings"][symbol] = {"shares": shares, "avg_price": price}
            trade = {"action": "BUY", "symbol": symbol, "shares": shares, "price": price, "time": timestamp}
            portfolio["trades"].append(trade)
            print(f"✅ BUY {shares} shares of {symbol} at ${price}")
        else:
            print(f"❌ Not enough cash to buy {shares} shares of {symbol}")

    elif decision == "SELL" and shares > 0:
        if symbol in portfolio["holdings"] and portfolio["holdings"][symbol]["shares"] >= shares:
            portfolio["cash"] += shares * price
            portfolio["cash"] = round(portfolio["cash"], 2)
            portfolio["holdings"][symbol]["shares"] -= shares
            if portfolio["holdings"][symbol]["shares"] == 0:
                del portfolio["holdings"][symbol]
            trade = {"action": "SELL", "symbol": symbol, "shares": shares, "price": price, "time": timestamp}
            portfolio["trades"].append(trade)
            print(f"✅ SELL {shares} shares of {symbol} at ${price}")
        else:
            print(f"❌ Not enough shares to sell")
    else:
        print(f"⏸️  HOLD {symbol}")

    return trade

def log_to_obsidian(portfolio, symbol, decision, shares, price, reasoning, stock_data):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    filename = f"{VAULT_PATH}/Trades/{date_str}-{symbol}.md"
    content = f"""# {symbol} — {decision} — {date_str}

## Stock Data
- **Price:** ${price}
- **Change:** {stock_data['change_pct']}%
- **Volume:** {stock_data['volume']}
- **Time:** {time_str}

## AI Decision
**Action:** {decision}
**Shares:** {shares}

## Claude's Reasoning
{reasoning}

## Portfolio After Trade
- **Cash:** ${round(portfolio['cash'], 2)}
- **Holdings:** {json.dumps(portfolio['holdings'], indent=2)}
"""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"📝 Logged to Obsidian: {filename}")

def run_trading_session():
    portfolio = load_portfolio()

    print(f"\n{'='*50}")
    print(f"🤖 FridayTrader Session — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Cash: ${portfolio['cash']} | Holdings: {list(portfolio['holdings'].keys())}")
    print(f"{'='*50}\n")

    for symbol in WATCHLIST:
        print(f"\n📊 Analyzing {symbol}...")
        stock_data = get_stock_data(symbol)
        if not stock_data:
            print(f"⚠️  Could not fetch data for {symbol}")
            continue

        print(f"   Price: ${stock_data['price']} | Change: {stock_data['change_pct']}%")
        response = ask_claude(stock_data, portfolio)
        decision, shares, reasoning = parse_decision(response)

        print(f"   Claude says: {decision} {shares} shares")
        print(f"   Reason: {reasoning}")

        execute_trade(portfolio, decision, shares, symbol, stock_data['price'])
        log_to_obsidian(portfolio, symbol, decision, shares, stock_data['price'], reasoning, stock_data)

    total_value = portfolio['cash']
    for symbol, data in portfolio['holdings'].items():
        stock = get_stock_data(symbol)
        if stock:
            total_value += data['shares'] * stock['price']

    save_portfolio(portfolio)

    print(f"\n{'='*50}")
    print(f"📈 Session Complete!")
    print(f"💰 Cash: ${round(portfolio['cash'], 2)}")
    print(f"📊 Total Portfolio Value: ${round(total_value, 2)}")
    print(f"📉 P&L: ${round(total_value - STARTING_CASH, 2)}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    run_trading_session()
