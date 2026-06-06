
import anthropic
import yfinance as yf
import json
import os
import schedule
import time
from datetime import datetime
import pytz
import numpy as np
import pandas as pd

from config import ADVANCED_LAYER_WEIGHTS, ADVANCED_CAP
from advanced_strategies import find_cointegrated_pairs, train_ml_model, combine_signals

STARTING_CASH = 10000.00
WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "NNE", "LEU", "RDW", "DCO", "CEG", "CCJ", "COST", "TJX", "CAVA", "CMG", "VRT", "CRDO"]
VAULT = os.path.expanduser("~/Documents/FridayTrader")
PORTFOLIO_FILE = os.path.join(VAULT, "portfolio.json")
PERFORMANCE_FILE = os.path.join(VAULT, "performance.json")
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
MAX_POSITION_PCT = 0.20
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 70

client = anthropic.Anthropic()

# ── Advanced signal cache (rebuilt at most once per calendar day) ─────────────
_PRICE_CACHE = {"df": None, "date": None}
_ADV_CACHE   = {"model": None, "pairs": None, "date": None}

def get_price_frame():
    today = datetime.now().strftime("%Y-%m-%d")
    if _PRICE_CACHE["df"] is not None and _PRICE_CACHE["date"] == today:
        return _PRICE_CACHE["df"]
    print("  [adv] Downloading 2y price history (cached daily)...")
    raw = yf.download(WATCHLIST, period="2y", auto_adjust=True, progress=False)
    df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    df = df.dropna(how="all")
    _PRICE_CACHE["df"] = df
    _PRICE_CACHE["date"] = today
    return df

def get_advanced_cache(frame):
    today = datetime.now().strftime("%Y-%m-%d")
    if _ADV_CACHE["date"] == today and _ADV_CACHE["model"] is not None:
        return _ADV_CACHE["model"], _ADV_CACHE["pairs"]
    print("  [adv] Training ML model + finding pairs (once per day)...")
    model, _ = train_ml_model(frame, verbose=True)
    pairs = find_cointegrated_pairs(frame)
    _ADV_CACHE["model"] = model
    _ADV_CACHE["pairs"] = pairs
    _ADV_CACHE["date"] = today
    return model, pairs

def _fmt_adv_detail(symbol, adv_detail):
    """Format per-symbol advanced signal breakdown for Obsidian logging."""
    if not adv_detail:
        return "none"
    reg = adv_detail.get("regime", {}).get(symbol, {})
    ml  = adv_detail.get("ml", {}).get(symbol, {})
    pair_str = next(
        (f"{k}(z={v['z']:+.1f})" for k, v in adv_detail.get("pairs", {}).items() if symbol in k),
        "none"
    )
    reg_str = (f"{reg.get('regime','?')}({reg.get('score', 0):+d} {reg.get('strategy','?')})"
               if reg else "none")
    ml_str  = (f"prob_up={ml.get('prob_up', 0):.2f}({ml.get('score', 0):+d})"
               if ml else "none")
    return f"regime={reg_str} | ml={ml_str} | pairs={pair_str}"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "holdings": {}, "trades": [], "sessions": 0}

def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)

def load_performance():
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE) as f:
            return json.load(f)
    return {"snapshots": [], "win_trades": 0, "loss_trades": 0, "total_realized_pnl": 0.0, "sp500_start": None}

def save_performance(perf):
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(perf, f, indent=2)

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def calculate_macd(prices):
    if len(prices) < 35:
        return 0, 0, 0
    prices = np.array(prices, dtype=float)
    def ema(data, period):
        k = 2 / (period + 1)
        e = [data[0]]
        for p in data[1:]:
            e.append(p * k + e[-1] * (1 - k))
        return np.array(e)
    macd_line = ema(prices, 12) - ema(prices, 26)
    signal_line = ema(macd_line, 9)
    histogram = macd_line - signal_line
    return round(float(macd_line[-1]), 4), round(float(signal_line[-1]), 4), round(float(histogram[-1]), 4)

def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="60d")
        if hist.empty or len(hist) < 30:
            return None
        closes = hist["Close"].values
        volumes = hist["Volume"].values
        current = round(float(closes[-1]), 2)
        prev = round(float(closes[-2]), 2)
        rsi = calculate_rsi(closes)
        macd, macd_sig, macd_hist = calculate_macd(closes)
        trend_5d = round((closes[-1] - closes[-5]) / closes[-5] * 100, 2) if len(closes) >= 5 else 0
        vol_ratio = round(volumes[-1] / np.mean(volumes[-20:]), 2) if len(volumes) >= 20 else 1.0
        recent_high = round(float(np.max(closes[-20:])), 2)
        recent_low = round(float(np.min(closes[-20:])), 2)
        price_range = recent_high - recent_low
        price_pos = round((current - recent_low) / price_range * 100, 1) if price_range > 0 else 50
        return {
            "symbol": symbol, "price": current, "change_pct": round((current-prev)/prev*100,2),
            "volume": int(volumes[-1]), "volume_ratio": vol_ratio,
            "rsi": rsi, "macd_histogram": macd_hist, "macd": macd, "macd_signal": macd_sig,
            "trend_5d": trend_5d, "recent_high": recent_high, "recent_low": recent_low, "price_position": price_pos
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None

def get_sp500_change():
    try:
        sp = yf.Ticker("^GSPC")
        hist = sp.history(period="5d")
        if len(hist) >= 2:
            change = (hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
            return round(float(change), 2), round(float(hist["Close"].iloc[-1]), 2)
    except:
        pass
    return 0.0, None

def get_headlines(symbol):
    try:
        import urllib.request, re
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            content = r.read().decode("utf-8")
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", content)[1:5]
        return titles
    except:
        return []

def evaluate_signals(data):
    signals, score = [], 0
    if data["rsi"] < RSI_OVERSOLD:
        signals.append(f"RSI={data['rsi']} OVERSOLD (bullish)"); score += 2
    elif data["rsi"] > RSI_OVERBOUGHT:
        signals.append(f"RSI={data['rsi']} OVERBOUGHT (avoid buying)"); score -= 2
    else:
        signals.append(f"RSI={data['rsi']} neutral")
    if data["macd_histogram"] > 0:
        signals.append(f"MACD hist={data['macd_histogram']} bullish"); score += 2
    else:
        signals.append(f"MACD hist={data['macd_histogram']} bearish"); score -= 1
    if data["volume_ratio"] >= 1.5:
        signals.append(f"Volume {data['volume_ratio']}x — strong"); score += 1
    elif data["volume_ratio"] < 0.8:
        signals.append(f"Volume {data['volume_ratio']}x — weak"); score -= 1
    if data["trend_5d"] > 2:
        signals.append(f"5d trend +{data['trend_5d']}% uptrend"); score += 2
    elif data["trend_5d"] < -2:
        signals.append(f"5d trend {data['trend_5d']}% downtrend"); score -= 2
    if data["price_position"] < 30:
        signals.append(f"Near 20d low — support zone"); score += 1
    elif data["price_position"] > 80:
        signals.append(f"Near 20d high — resistance"); score -= 1
    return signals, score

def check_stop_take(portfolio, data):
    sym = data["symbol"]
    if sym not in portfolio["holdings"]:
        return None, None
    h = portfolio["holdings"][sym]
    pnl_pct = (data["price"] - h["avg_price"]) / h["avg_price"]
    if pnl_pct <= -STOP_LOSS_PCT:
        return "SELL", f"STOP-LOSS: {round(pnl_pct*100,2)}% loss"
    if pnl_pct >= TAKE_PROFIT_PCT:
        return "SELL", f"TAKE-PROFIT: +{round(pnl_pct*100,2)}% gain"
    return None, None

def ask_claude(data, portfolio, headlines, signals, score, sp500):
    total = portfolio["cash"] + sum(h["shares"]*h["avg_price"] for h in portfolio["holdings"].values())
    max_shares = int(total * MAX_POSITION_PCT / data["price"])
    held = portfolio["holdings"].get(data["symbol"])
    held_info = f"Holding {held['shares']} shares @ avg ${held['avg_price']}" if held else "No position"
    bias = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"
    news = chr(10).join([f"- {h}" for h in headlines]) if headlines else "- None available"
    sigs = chr(10).join([f"- {s}" for s in signals])
    prompt = f"""Analyze {data["symbol"]} and decide BUY/SELL/HOLD.

PRICE: ${data["price"]} ({data["change_pct"]:+.2f}% today) | 5d trend: {data["trend_5d"]:+.2f}%
RSI: {data["rsi"]} | MACD histogram: {data["macd_histogram"]} | Volume: {data["volume_ratio"]}x avg
Price position: {data["price_position"]}% of 20d range (0=low, 100=high)
Score: {score} (blended) — {bias}

Signals:
{sigs}

News:
{news}

S&P 500 today: {sp500:+.2f}%
Cash: ${portfolio["cash"]:,.2f} | {held_info} | Max buy: {max_shares} shares

Rules: Only BUY if RSI<70 AND MACD hist>0 AND volume>0.8x. SELL if stop-loss/take-profit hit.

Reply EXACTLY:
DECISION: [BUY/SELL/HOLD]
SHARES: [number]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [2 sentences]"""
    try:
        r = client.messages.create(model="claude-sonnet-4-6", max_tokens=200, messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    except Exception as e:
        return f"DECISION: HOLD\nSHARES: 0\nCONFIDENCE: LOW\nREASONING: API error {e}"

def parse_decision(response):
    decision, shares, confidence, reasoning = "HOLD", 0, "MEDIUM", ""
    for line in response.strip().split("\n"):
        if line.startswith("DECISION:"): decision = line.split(":",1)[1].strip().upper()
        elif line.startswith("SHARES:"):
            try: shares = int(line.split(":",1)[1].strip())
            except: shares = 0
        elif line.startswith("CONFIDENCE:"): confidence = line.split(":",1)[1].strip()
        elif line.startswith("REASONING:"): reasoning = line.split(":",1)[1].strip()
    return decision, shares, confidence, reasoning

def execute_trade(portfolio, decision, shares, symbol, price, reasoning):
    perf = load_performance()
    if decision == "BUY" and shares > 0:
        cost = shares * price
        if cost > portfolio["cash"]:
            shares = int(portfolio["cash"] / price)
            cost = shares * price
        if shares == 0:
            print(f"  Not enough cash"); return
        portfolio["cash"] = round(portfolio["cash"] - cost, 2)
        if symbol in portfolio["holdings"]:
            h = portfolio["holdings"][symbol]
            total_s = h["shares"] + shares
            h["avg_price"] = round((h["shares"]*h["avg_price"] + cost) / total_s, 2)
            h["shares"] = total_s
        else:
            portfolio["holdings"][symbol] = {"shares": shares, "avg_price": price}
        portfolio["trades"].append({"action":"BUY","symbol":symbol,"shares":shares,"price":price,"time":datetime.now().strftime("%Y-%m-%d %H:%M"),"reasoning":reasoning[:100]})
        print(f"  ✅ BUY {shares} shares of {symbol} @ ${price}")
    elif decision == "SELL" and symbol in portfolio["holdings"]:
        h = portfolio["holdings"][symbol]
        proceeds = round(h["shares"] * price, 2)
        pnl = round(proceeds - h["shares"]*h["avg_price"], 2)
        portfolio["cash"] = round(portfolio["cash"] + proceeds, 2)
        del portfolio["holdings"][symbol]
        perf["total_realized_pnl"] = round(perf.get("total_realized_pnl",0) + pnl, 2)
        if pnl >= 0: perf["win_trades"] = perf.get("win_trades",0) + 1
        else: perf["loss_trades"] = perf.get("loss_trades",0) + 1
        portfolio["trades"].append({"action":"SELL","symbol":symbol,"shares":h["shares"],"price":price,"time":datetime.now().strftime("%Y-%m-%d %H:%M"),"realized_pnl":pnl})
        save_performance(perf)
        print(f"  ✅ SELL {h['shares']} shares of {symbol} @ ${price} | P&L: ${pnl:+.2f}")

def log_to_obsidian(symbol, decision, shares, price, confidence, reasoning, data, signals, score, adv_detail=None):
    trades_dir = os.path.join(VAULT, "Trades")
    os.makedirs(trades_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    bias = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"
    adv_line = _fmt_adv_detail(symbol, adv_detail)
    content = f"""# {symbol} — {decision} — {date_str}
**Action:** {decision} {shares} shares @ ${price} | **Confidence:** {confidence}

| Indicator | Value | Signal |
|-----------|-------|--------|
| RSI | {data["rsi"]} | {"OVERSOLD" if data["rsi"]<35 else "OVERBOUGHT" if data["rsi"]>70 else "Neutral"} |
| MACD Histogram | {data["macd_histogram"]} | {"Bullish" if data["macd_histogram"]>0 else "Bearish"} |
| 5-Day Trend | {data["trend_5d"]}% | {"Up" if data["trend_5d"]>0 else "Down"} |
| Volume Ratio | {data["volume_ratio"]}x | {"High" if data["volume_ratio"]>1.2 else "Low" if data["volume_ratio"]<0.8 else "Normal"} |
| Price Position | {data["price_position"]}% | {"Near Low" if data["price_position"]<30 else "Near High" if data["price_position"]>80 else "Mid"} |

**Score: {score} (blended) — {bias}**
**Advanced:** {adv_line}
**Reasoning:** {reasoning}
"""
    with open(os.path.join(trades_dir, f"{date_str}-{symbol}-v3.md"), "w") as f:
        f.write(content)
    print(f"  📝 Logged to Obsidian")

def run_trading_session():
    portfolio = load_portfolio()
    portfolio["sessions"] = portfolio.get("sessions", 0) + 1
    sp500_change, sp500_price = get_sp500_change()
    print(f"\n{'='*55}")
    print(f"🤖 FridayTrader v3 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 S&P 500: {sp500_change:+.2f}% | Cash: ${portfolio['cash']:,.2f}")
    print(f"Holdings: {list(portfolio['holdings'].keys())}")
    print(f"{'='*55}\n")
    # ── Build advanced signals once for this session ──────────────────────────
    adv_combined, adv_detail = {}, {}
    try:
        frame = get_price_frame()
        adv_model, adv_pairs = get_advanced_cache(frame)
        adv_combined, adv_detail = combine_signals(
            frame, weights=ADVANCED_LAYER_WEIGHTS, ml_model=adv_model, pairs=adv_pairs)
        print(f"  [adv] Advanced signals ready for {len(adv_combined)} symbols\n")
    except Exception as e:
        print(f"  [adv] Advanced layer error (proceeding without): {e}\n")
    # ──────────────────────────────────────────────────────────────────────────

    current_prices = {}
    for symbol in WATCHLIST:
        print(f"\n📊 Analyzing {symbol}...")
        data = get_stock_data(symbol)
        if not data:
            continue
        current_prices[symbol] = data["price"]
        print(f"  ${data['price']} ({data['change_pct']:+.2f}%) | RSI:{data['rsi']} | MACD:{data['macd_histogram']} | 5d:{data['trend_5d']:+.2f}% | Vol:{data['volume_ratio']}x")
        auto_dec, auto_reason = check_stop_take(portfolio, data)
        if auto_dec:
            print(f"  🚨 Auto-{auto_dec}: {auto_reason}")
            execute_trade(portfolio, auto_dec, 0, symbol, data["price"], auto_reason)
            signals, score = evaluate_signals(data)
            log_to_obsidian(symbol, auto_dec, 0, data["price"], "HIGH", auto_reason, data, signals, score, adv_detail)
            continue
        signals, score = evaluate_signals(data)

        # Blend advanced score — cap contribution so it nudges, never dominates
        adv_raw    = adv_combined.get(symbol, 0)
        adv_contrib = max(-ADVANCED_CAP, min(ADVANCED_CAP, adv_raw))
        blended_score = score + adv_contrib
        adv_line = _fmt_adv_detail(symbol, adv_detail)
        print(f"  [adv] {symbol}: tech={score:+d}, adv={adv_contrib:+d} → blended={blended_score:+d} | {adv_line}")

        headlines = get_headlines(symbol)
        print(f"  🧠 Asking Claude (blended score: {blended_score})...")
        response = ask_claude(data, portfolio, headlines, signals, blended_score, sp500_change)
        decision, shares, confidence, reasoning = parse_decision(response)
        print(f"  Decision: {decision} {shares} shares ({confidence})")
        print(f"  {reasoning[:80]}")
        if decision in ("BUY", "SELL"):
            execute_trade(portfolio, decision, shares, symbol, data["price"], reasoning)
        else:
            print(f"  ⏸️  HOLD")
        log_to_obsidian(symbol, decision, shares, data["price"], confidence, reasoning, data, signals, blended_score, adv_detail)
    total = portfolio["cash"] + sum(h["shares"]*current_prices.get(s, h["avg_price"]) for s,h in portfolio["holdings"].items())
    pnl = total - STARTING_CASH
    perf = load_performance()
    if not perf.get("sp500_start") and sp500_price:
        perf["sp500_start"] = sp500_price
    perf["snapshots"].append({"date":datetime.now().strftime("%Y-%m-%d %H:%M"),"value":round(total,2),"cash":round(portfolio["cash"],2),"pnl":round(pnl,2)})
    save_performance(perf)
    save_portfolio(portfolio)
    total_closed = perf.get("win_trades",0) + perf.get("loss_trades",0)
    wr = perf["win_trades"]/total_closed*100 if total_closed > 0 else 0
    print(f"\n{'='*55}")
    print(f"📈 Session Complete! Portfolio: ${total:,.2f} | P&L: ${pnl:+,.2f} ({pnl/STARTING_CASH*100:+.2f}%)")
    print(f"🏆 Win Rate: {wr:.1f}% | Cash: ${portfolio['cash']:,.2f}")
    print(f"{'='*55}\n")

def run_scheduler():
    print("🚀 FridayTrader v3 — RSI + MACD + Volume + 5-Day Trend")
    print("📅 Runs Mon-Fri at 22:00 and 02:00 SGT\n")
    run_trading_session()
    for day in ["monday","tuesday","wednesday","thursday","friday"]:
        getattr(schedule.every(), day).at("22:00").do(run_trading_session)
        getattr(schedule.every(), day).at("02:00").do(run_trading_session)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "schedule"
    if cmd == "once": run_trading_session()
    elif cmd == "schedule": run_scheduler()
