
import anthropic
import yfinance as yf
import json
import os
import functools
import schedule
import time
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from config import (ADVANCED_LAYER_WEIGHTS, ADVANCED_CAP,
                    ATR_STOP_MULT, ATR_TARGET_MULT,
                    DRAWDOWN_CIRCUIT_BREAKER, DRAWDOWN_RESUME_PCT,
                    SPY_REGIME_WINDOW, EARNINGS_BLACKOUT_DAYS, EARNINGS_EXIT_DAYS,
                    WEEKLY_RSI_PERIOD, WEEKLY_RSI_MAX,
                    RSI_OVERSOLD, RSI_DEEP_OVERSOLD, RSI_OVERBOUGHT,
                    VOLUME_SURGE_MIN, RS3M_BULL_THRESH, RS3M_BEAR_THRESH,
                    MIN_BUY_SCORE, MAX_SELL_SCORE)
from advanced_strategies import find_cointegrated_pairs, train_ml_model, combine_signals

STARTING_CASH = 10000.00
WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "NNE", "LEU", "RDW", "DCO", "CEG", "CCJ", "COST", "TJX", "CAVA", "CMG", "VRT", "CRDO"]
VAULT = os.path.expanduser("~/Documents/FridayTrader")
PORTFOLIO_FILE = os.path.join(VAULT, "portfolio.json")
PERFORMANCE_FILE = os.path.join(VAULT, "performance.json")
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
MAX_POSITION_PCT = 0.20

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

client = anthropic.Anthropic()

# ── Exponential-backoff retry decorator ───────────────────────────────────────
def _retry(max_retries=3, delays=(2, 4, 8), reraise=False):
    """Retry on any exception with fixed backoff. Returns None (or reraises) after exhausting retries."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt, delay in enumerate(delays[:max_retries], 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        print(f"  [retry] {fn.__name__} failed after {max_retries} attempts: {e}")
                        if reraise:
                            raise
                        return None
                    print(f"  [retry] {fn.__name__} attempt {attempt} failed ({e}), retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@_retry(max_retries=3, delays=(2, 4, 8), reraise=False)
def _fetch_history(symbol, period="70d"):
    return yf.Ticker(symbol).history(period=period)

@_retry(max_retries=3, delays=(2, 4, 8), reraise=False)
def _fetch_history_weekly(symbol, period="6mo"):
    return yf.Ticker(symbol).history(period=period, interval="1wk")

@_retry(max_retries=3, delays=(2, 4, 8), reraise=False)
def _fetch_sp500_history():
    return yf.Ticker("^GSPC").history(period="5d")

@_retry(max_retries=3, delays=(2, 4, 8), reraise=True)
def _batch_download(tickers, period, **kwargs):
    return yf.download(tickers, period=period, **kwargs)

# ── SPY 200-day regime filter ─────────────────────────────────────────────────

def get_spy_regime():
    """Return (is_bull, spy_close, spy_ma200). Defaults to bull on error to avoid false blocks."""
    try:
        hist = _fetch_history("SPY", "1y")
        if hist is None or len(hist) < SPY_REGIME_WINDOW:
            return True, None, None
        closes = hist["Close"].values.astype(float)
        spy_close = round(float(closes[-1]), 2)
        spy_ma200 = round(float(np.mean(closes[-SPY_REGIME_WINDOW:])), 2)
        return bool(spy_close > spy_ma200), spy_close, spy_ma200
    except Exception as e:
        print(f"  [regime] SPY 200MA check failed ({e}), defaulting to BULL")
        return True, None, None


# ── Earnings blackout ─────────────────────────────────────────────────────────

def get_earnings_date(symbol):
    """Return next earnings date as datetime.date, or None if unavailable."""
    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None:
            return None
        # yfinance returns DataFrame or dict depending on version
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
            elif len(cal) > 0:
                val = cal.iloc[0, 0]
            else:
                return None
        elif isinstance(cal, dict):
            val = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(val, (list, np.ndarray)):
                val = val[0] if len(val) > 0 else None
        else:
            return None
        if val is None or (hasattr(val, "__class__") and pd.isna(val)):
            return None
        return pd.Timestamp(val).date()
    except Exception:
        return None


def earnings_blackout_check(symbol, earnings_date):
    """Return (skip_entirely, force_sell, days_to_earnings)."""
    if earnings_date is None:
        return False, False, None
    today = datetime.now().date()
    days = (earnings_date - today).days
    return days <= EARNINGS_BLACKOUT_DAYS, days <= EARNINGS_EXIT_DAYS, days


# ── Multi-timeframe weekly signal ─────────────────────────────────────────────

def get_weekly_signal(symbol):
    """Return (weekly_bullish, weekly_rsi, weekly_macd_hist). Defaults True on error."""
    try:
        hist = _fetch_history_weekly(symbol)
        if hist is None or len(hist) < 15:
            return True, None, None
        closes = hist["Close"].values.astype(float)
        w_rsi = calculate_rsi(closes, period=WEEKLY_RSI_PERIOD)
        _, _, w_macd_hist = calculate_macd(closes)
        weekly_bullish = bool(w_rsi < WEEKLY_RSI_MAX and w_macd_hist > 0)
        return weekly_bullish, round(w_rsi, 2), round(w_macd_hist, 4)
    except Exception as e:
        print(f"  [weekly] {symbol} weekly signal failed ({e}), defaulting to bullish")
        return True, None, None


# ── Telegram alerts ───────────────────────────────────────────────────────────

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import urllib.request as _ur
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": msg}).encode()
        req = _ur.Request(url, data=data, headers={"Content-Type": "application/json"})
        with _ur.urlopen(req, timeout=5):
            pass
    except Exception as e:
        print(f"  [telegram] Failed to send alert: {e}")


# ── Advanced signal cache (rebuilt at most once per calendar day) ─────────────
_PRICE_CACHE = {"df": None, "date": None}
_ADV_CACHE   = {"model": None, "pairs": None, "date": None}

def get_price_frame():
    today = datetime.now().strftime("%Y-%m-%d")
    if _PRICE_CACHE["df"] is not None and _PRICE_CACHE["date"] == today:
        return _PRICE_CACHE["df"]
    print("  [adv] Downloading 2y price history (cached daily)...")
    raw = _batch_download(WATCHLIST, period="2y", auto_adjust=True, progress=False)
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

def _fmt_adv_detail(symbol, adv_detail, adv_contrib=None):
    """Format per-symbol advanced signal breakdown for Obsidian logging.
    Example: regime=TREND(+2) | ml=0.61(+1) | pairs=none | adv_total=+2
    """
    if not adv_detail:
        return "none"
    reg = adv_detail.get("regime", {}).get(symbol, {})
    ml  = adv_detail.get("ml", {}).get(symbol, {})
    pair_str = next(
        (f"{k}(z={v['z']:+.1f})" for k, v in adv_detail.get("pairs", {}).items() if symbol in k),
        "none"
    )
    reg_str = (f"{reg.get('regime','?')}({reg.get('score', 0):+d})"
               if reg else "none")
    ml_str  = (f"{ml.get('prob_up', 0):.2f}({ml.get('score', 0):+d})"
               if ml else "none")
    total_str = f" | adv_total={adv_contrib:+d}" if adv_contrib is not None else ""
    return f"regime={reg_str} | ml={ml_str} | pairs={pair_str}{total_str}"

def save_signals(session_signals):
    path = os.path.join(VAULT, "signals.json")
    with open(path, "w") as f:
        json.dump({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "signals": session_signals}, f, indent=2)

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            p = json.load(f)
        p.setdefault("peak_value", STARTING_CASH)
        p.setdefault("drawdown_halted", False)
        p.setdefault("spy_regime", "BULL")
        p.setdefault("spy_close", None)
        p.setdefault("spy_ma200", None)
        return p
    return {"cash": STARTING_CASH, "holdings": {}, "trades": [], "sessions": 0,
            "peak_value": STARTING_CASH, "drawdown_halted": False,
            "spy_regime": "BULL", "spy_close": None, "spy_ma200": None}

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

def calculate_atr(hist, period=14):
    """Wilder's ATR-14 from a yfinance history DataFrame."""
    if len(hist) < period + 1:
        return None
    high  = hist["High"].values.astype(float)
    low   = hist["Low"].values.astype(float)
    close = hist["Close"].values.astype(float)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]),
                   np.abs(low[1:]  - close[:-1]))
    )
    atr = np.empty(len(tr))
    atr[0] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return round(float(atr[-1]), 4)

def get_stock_data(symbol):
    try:
        hist = _fetch_history(symbol, "70d")
        if hist is None or hist.empty or len(hist) < 30:
            return None
        closes = hist["Close"].values.astype(float)
        volumes = hist["Volume"].values.astype(float)
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
        atr = calculate_atr(hist)
        ma50 = round(float(np.mean(closes[-50:])), 2) if len(closes) >= 50 else None
        return_3m = round((closes[-1] - closes[-63]) / closes[-63] * 100, 2) if len(closes) >= 63 else None
        return {
            "symbol": symbol, "price": current, "change_pct": round((current-prev)/prev*100,2),
            "volume": int(volumes[-1]), "volume_ratio": vol_ratio,
            "rsi": rsi, "macd_histogram": macd_hist, "macd": macd, "macd_signal": macd_sig,
            "trend_5d": trend_5d, "recent_high": recent_high, "recent_low": recent_low,
            "price_position": price_pos, "atr": atr, "ma50": ma50, "return_3m": return_3m,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None

def get_sp500_change():
    try:
        hist = _fetch_sp500_history()
        if hist is not None and len(hist) >= 2:
            change = (hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
            return round(float(change), 2), round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
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

def evaluate_signals(data, spy3m_return=None):
    """Winning signal stack: RSI<35 primary gate + volume surge confirmation.
    OOS ablation confirmed this is the only combination with positive Sharpe (+1.68).
    50MA and RS3M are logged as informational but do NOT affect the score —
    they were blocking RSI+volume trades and hurt OOS performance.
    Returns (signals, score, rsi_gate) where rsi_gate=True means RSI<35 gate passes."""
    signals, score = [], 0

    # RSI — PRIMARY GATE + score. Hard-block BUY when RSI >= RSI_OVERSOLD.
    rsi = data["rsi"]
    rsi_gate = rsi < RSI_OVERSOLD
    if rsi < RSI_DEEP_OVERSOLD:
        signals.append(f"RSI={rsi} DEEPLY OVERSOLD — strong entry signal"); score += 3
    elif rsi < RSI_OVERSOLD:
        signals.append(f"RSI={rsi} OVERSOLD — primary gate passes"); score += 2
    elif rsi > RSI_OVERBOUGHT:
        signals.append(f"RSI={rsi} OVERBOUGHT — avoid buying"); score -= 2
    else:
        signals.append(f"RSI={rsi} neutral (gate BLOCKED — BUY requires RSI<{RSI_OVERSOLD})")

    # Volume surge — only confirmed confirming signal from OOS ablation
    if data["volume_ratio"] >= VOLUME_SURGE_MIN:
        signals.append(f"Volume {data['volume_ratio']}x 20d avg — surge confirmation"); score += 2
    elif data["volume_ratio"] < 0.8:
        signals.append(f"Volume {data['volume_ratio']}x — low conviction"); score -= 1
    else:
        signals.append(f"Volume {data['volume_ratio']}x — normal")

    # Price position in 20d range — contextual, small weight
    if data["price_position"] < 30:
        signals.append("Near 20d low — support zone"); score += 1
    elif data["price_position"] > 80:
        signals.append("Near 20d high — resistance zone"); score -= 1

    # 50MA — INFORMATIONAL ONLY (not scored: was blocking RSI+volume trades)
    ma50 = data.get("ma50")
    if ma50 is not None:
        ma50_rel = "above" if data["price"] > ma50 else "below"
        signals.append(f"50MA ${ma50} [{ma50_rel}] — info only, not scored")

    # 3m RS vs SPY — INFORMATIONAL ONLY (not scored: hurt OOS performance)
    rs3m = None
    if data.get("return_3m") is not None and spy3m_return is not None:
        rs3m = round(data["return_3m"] - spy3m_return, 2)
        signals.append(f"3m RS vs SPY: {rs3m:+.1f}% — info only, not scored")
    data["rs3m"] = rs3m
    return signals, score, rsi_gate

def check_stop_take(portfolio, data):
    sym = data["symbol"]
    if sym not in portfolio["holdings"]:
        return None, None
    h = portfolio["holdings"][sym]
    entry = h["avg_price"]
    price = data["price"]
    atr   = h.get("atr")  # stored at entry time; None for legacy holdings
    if atr is not None:
        stop_price   = entry - atr * ATR_STOP_MULT
        target_price = entry + atr * ATR_TARGET_MULT
        pnl_pct = (price - entry) / entry * 100
        if price <= stop_price:
            return "SELL", f"ATR-STOP: {pnl_pct:+.2f}% (stop=${stop_price:.2f}, ATR={atr:.2f})"
        if price >= target_price:
            return "SELL", f"ATR-TARGET: {pnl_pct:+.2f}% (target=${target_price:.2f}, ATR={atr:.2f})"
    else:
        # Legacy percentage stops for holdings entered before ATR support
        pnl_pct = (price - entry) / entry
        if pnl_pct <= -STOP_LOSS_PCT:
            return "SELL", f"STOP-LOSS: {pnl_pct*100:+.2f}% loss"
        if pnl_pct >= TAKE_PROFIT_PCT:
            return "SELL", f"TAKE-PROFIT: {pnl_pct*100:+.2f}% gain"
    return None, None

def ask_claude(data, portfolio, headlines, signals, score, sp500, rsi_gate=True,
               spy_regime=True, weekly_bullish=True):
    total = portfolio["cash"] + sum(h["shares"]*h["avg_price"] for h in portfolio["holdings"].values())
    max_shares = int(total * MAX_POSITION_PCT / data["price"])
    held = portfolio["holdings"].get(data["symbol"])
    held_info = f"Holding {held['shares']} shares @ avg ${held['avg_price']}" if held else "No position"
    bias = "BULLISH" if score >= MIN_BUY_SCORE else "BEARISH" if score <= MAX_SELL_SCORE else "NEUTRAL"
    news = chr(10).join([f"- {h}" for h in headlines]) if headlines else "- None available"
    sigs = chr(10).join([f"- {s}" for s in signals])
    ma50_str = f"${data['ma50']}" if data.get("ma50") else "N/A"
    rs3m_str = f"{data.get('rs3m', 0):+.1f}%" if data.get("rs3m") is not None else "N/A"
    gates_blocked = []
    if not rsi_gate:      gates_blocked.append(f"RSI={data['rsi']} >= {RSI_OVERSOLD}")
    if not spy_regime:    gates_blocked.append("SPY < 200MA (bear market)")
    if not weekly_bullish: gates_blocked.append("weekly signal bearish")
    gates_str = "BLOCKED: " + ", ".join(gates_blocked) if gates_blocked else "all clear"
    prompt = f"""Analyze {data["symbol"]} and decide BUY/SELL/HOLD.

PRICE: ${data["price"]} ({data["change_pct"]:+.2f}% today)
RSI: {data["rsi"]} | 50MA: {ma50_str} | Volume: {data["volume_ratio"]}x avg | 3m RS vs SPY: {rs3m_str}
Price position: {data["price_position"]}% of 20d range (0=low, 100=high)
Score: {score} (blended) — {bias}

BUY gates: {gates_str}

Signals:
{sigs}

News:
{news}

S&P 500 today: {sp500:+.2f}%
Cash: ${portfolio["cash"]:,.2f} | {held_info} | Max buy: {max_shares} shares

Rules:
- BUY requires: RSI<{RSI_OVERSOLD} (primary gate), SPY>200MA, weekly signal bullish, score>={MIN_BUY_SCORE}
- RSI diagnostic: only RSI<35 has validated OOS edge (+8.2pp hit rate); MACD/trend have zero edge
- SELL if stop-loss/take-profit hit OR score<={MAX_SELL_SCORE}
- If any BUY gate is BLOCKED, reply HOLD

Reply EXACTLY:
DECISION: [BUY/SELL/HOLD]
SHARES: [number]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [2 sentences]"""
    for attempt, delay in enumerate([2, 4, 8], 1):
        try:
            r = client.messages.create(model="claude-sonnet-4-6", max_tokens=200,
                                       messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        except Exception as e:
            if attempt == 3:
                print(f"  [retry] Claude API failed after 3 attempts: {e}")
                return f"DECISION: HOLD\nSHARES: 0\nCONFIDENCE: LOW\nREASONING: API error after retries: {e}"
            print(f"  [retry] Claude API attempt {attempt} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)

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

def execute_trade(portfolio, decision, shares, symbol, price, reasoning, atr=None,
                  rsi=None, blended_score=None, spy_regime=True):
    perf = load_performance()
    regime_str = "BULL" if spy_regime else "BEAR"
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
            if atr is not None:
                h["atr"] = atr
        else:
            portfolio["holdings"][symbol] = {"shares": shares, "avg_price": price, "atr": atr}
        portfolio["trades"].append({"action":"BUY","symbol":symbol,"shares":shares,"price":price,"time":datetime.now().strftime("%Y-%m-%d %H:%M"),"reasoning":reasoning[:100]})
        print(f"  ✅ BUY {shares} shares of {symbol} @ ${price}")
        rsi_str = f" | RSI:{rsi}" if rsi is not None else ""
        score_str = f" | Score:{blended_score:+d}" if blended_score is not None else ""
        send_telegram(f"🟢 BUY {symbol} — {shares} shares @ ${price}{rsi_str} | Regime:{regime_str}{score_str}")
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
        emoji = "🟩" if pnl >= 0 else "🔴"
        send_telegram(f"{emoji} SELL {symbol} — {h['shares']} shares @ ${price} | P&L: ${pnl:+.2f} | {reasoning[:60]}")

def log_to_obsidian(symbol, decision, shares, price, confidence, reasoning, data, signals, score, adv_detail=None, adv_contrib=None):
    trades_dir = os.path.join(VAULT, "Trades")
    os.makedirs(trades_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    bias = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"
    adv_line = _fmt_adv_detail(symbol, adv_detail, adv_contrib)
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

    # ── 1. SPY 200-day regime filter ──────────────────────────────────────────
    spy_is_bull, spy_close, spy_ma200 = get_spy_regime()
    regime_label = "BULL" if spy_is_bull else "BEAR"
    portfolio["spy_regime"] = regime_label
    portfolio["spy_close"]  = spy_close
    portfolio["spy_ma200"]  = spy_ma200

    # ── 2. Compute SPY 3-month return (for RS signal) ─────────────────────────
    spy3m_return = None
    try:
        spy_hist = _fetch_history("SPY", "70d")
        if spy_hist is not None and len(spy_hist) >= 63:
            spy3m_return = round((float(spy_hist["Close"].iloc[-1]) - float(spy_hist["Close"].iloc[-63]))
                                  / float(spy_hist["Close"].iloc[-63]) * 100, 2)
    except Exception:
        pass

    print(f"\n{'='*55}")
    print(f"🤖 FridayTrader v3 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 S&P 500: {sp500_change:+.2f}% | Cash: ${portfolio['cash']:,.2f}")
    print(f"Holdings: {list(portfolio['holdings'].keys())}")
    spy_detail = f"${spy_close} vs 200MA ${spy_ma200}" if spy_close else "N/A"
    if spy_is_bull:
        print(f"  📈 Market regime: BULL ({spy_detail}) — normal operation")
    else:
        print(f"  🐻 Market regime: BEAR ({spy_detail}) — new BUYs HALTED")
        send_telegram(f"🐻 Market regime: BEAR — SPY ${spy_close} < 200MA ${spy_ma200}. BUYs halted.")
    if portfolio.get("drawdown_halted"):
        print(f"  ⚠️  Drawdown circuit breaker ACTIVE — new BUYs halted until recovery")
    print(f"{'='*55}\n")

    # ── 3. Build advanced signals once for this session ───────────────────────
    adv_combined, adv_detail = {}, {}
    try:
        frame = get_price_frame()
        adv_model, adv_pairs = get_advanced_cache(frame)
        adv_combined, adv_detail = combine_signals(
            frame, weights=ADVANCED_LAYER_WEIGHTS, ml_model=adv_model, pairs=adv_pairs)
        print(f"  [adv] Advanced signals ready for {len(adv_combined)} symbols\n")
    except Exception as e:
        print(f"  [adv] Advanced layer error (proceeding without): {e}\n")

    current_prices = {}
    session_signals = []

    for symbol in WATCHLIST:
        print(f"\n📊 Analyzing {symbol}...")

        # ── Earnings blackout check ───────────────────────────────────────────
        earnings_date = get_earnings_date(symbol)
        skip_buy, force_sell_earnings, days_to_earnings = earnings_blackout_check(symbol, earnings_date)
        if days_to_earnings is not None and days_to_earnings >= 0:
            print(f"  📅 Earnings in {days_to_earnings} day(s): {earnings_date}")

        data = get_stock_data(symbol)
        if not data:
            continue
        current_prices[symbol] = data["price"]
        ma50_str = f"50MA:${data['ma50']}" if data.get("ma50") else "50MA:N/A"
        rs3m_str = f"RS3m:{data['return_3m']:+.1f}%" if data.get("return_3m") is not None else ""
        print(f"  ${data['price']} ({data['change_pct']:+.2f}%) | RSI:{data['rsi']} | {ma50_str} | Vol:{data['volume_ratio']}x | {rs3m_str}")

        # ── Stop/take-profit check (overrides everything) ─────────────────────
        auto_dec, auto_reason = check_stop_take(portfolio, data)

        # ── Earnings forced exit ──────────────────────────────────────────────
        if not auto_dec and force_sell_earnings and symbol in portfolio["holdings"]:
            auto_dec = "SELL"
            auto_reason = f"Earnings blackout: earnings in {days_to_earnings} day(s) on {earnings_date}"
            print(f"  📅 Forced SELL: {auto_reason}")

        if auto_dec:
            print(f"  🚨 Auto-{auto_dec}: {auto_reason}")
            execute_trade(portfolio, auto_dec, 0, symbol, data["price"], auto_reason,
                          spy_regime=spy_is_bull)
            signals, score, rsi_gate = evaluate_signals(data, spy3m_return)
            log_to_obsidian(symbol, auto_dec, 0, data["price"], "HIGH", auto_reason,
                            data, signals, score, adv_detail)
            continue

        # ── Skip entirely if earnings blackout ────────────────────────────────
        if skip_buy and symbol not in portfolio["holdings"]:
            print(f"  ⛔ {symbol}: earnings blackout ({days_to_earnings} days to {earnings_date}) — skipping")
            continue

        # ── Compute signals ───────────────────────────────────────────────────
        signals, score, rsi_gate = evaluate_signals(data, spy3m_return)

        # ── Weekly MTF check ──────────────────────────────────────────────────
        weekly_bullish, w_rsi, w_macd_hist = get_weekly_signal(symbol)
        if not weekly_bullish and w_rsi is not None:
            print(f"  📉 Weekly: RSI={w_rsi} / MACD={w_macd_hist} — bearish (BUY blocked by MTF)")
        elif w_rsi is not None:
            print(f"  📈 Weekly: RSI={w_rsi} / MACD={w_macd_hist} — bullish")

        # ── Advanced layer blend ──────────────────────────────────────────────
        adv_raw     = adv_combined.get(symbol, 0)
        adv_contrib = max(-ADVANCED_CAP, min(ADVANCED_CAP, adv_raw))
        blended_score = score + adv_contrib
        adv_line = _fmt_adv_detail(symbol, adv_detail)
        print(f"  [adv] {symbol}: tech={score:+d}, adv={adv_contrib:+d} → blended={blended_score:+d} | {adv_line}")

        # ── Signal snapshot for dashboard ─────────────────────────────────────
        reg_d  = adv_detail.get("regime", {}).get(symbol, {})
        ml_d   = adv_detail.get("ml", {}).get(symbol, {})
        pair_d = adv_detail.get("pairs", {})
        pair_str = next((f"{k}(z={v['z']:+.1f})" for k, v in pair_d.items() if symbol in k), "none")
        session_signals.append({
            "sym": symbol, "price": data["price"],
            "rsi": data["rsi"], "ma50": data.get("ma50"),
            "volume_ratio": data.get("volume_ratio"),
            "rs3m": data.get("rs3m"),
            "tech_score": score,
            "rsi_gate": rsi_gate,
            "weekly_rsi": w_rsi, "weekly_macd": w_macd_hist, "weekly_bullish": weekly_bullish,
            "earnings_date": str(earnings_date) if earnings_date else None,
            "earnings_days": days_to_earnings,
            "regime": reg_d.get("regime", "UNKNOWN"),
            "regime_score": reg_d.get("score", 0),
            "ml_prob": ml_d.get("prob_up"),
            "ml_score": ml_d.get("score", 0),
            "pairs": pair_str,
            "adv_contrib": adv_contrib,
            "blended_score": blended_score,
        })

        headlines = get_headlines(symbol)
        print(f"  🧠 Asking Claude (blended score: {blended_score})...")
        response = ask_claude(data, portfolio, headlines, signals, blended_score, sp500_change,
                              rsi_gate=rsi_gate, spy_regime=spy_is_bull, weekly_bullish=weekly_bullish)
        decision, shares, confidence, reasoning = parse_decision(response)
        print(f"  Decision: {decision} {shares} shares ({confidence})")
        print(f"  {reasoning[:80]}")

        # ── Hard-gate enforcement (server-side, overrides Claude) ─────────────
        if decision == "BUY":
            if not rsi_gate:
                print(f"  🚫 BUY blocked: RSI={data['rsi']} >= {RSI_OVERSOLD} (primary gate)")
                send_telegram(f"⚠️ {symbol}: Claude said BUY but RSI={data['rsi']} blocked (gate requires <{RSI_OVERSOLD})")
                decision = "HOLD"
            elif not spy_is_bull:
                print(f"  🚫 BUY blocked: SPY in BEAR regime (SPY ${spy_close} < 200MA ${spy_ma200})")
                decision = "HOLD"
            elif not weekly_bullish:
                print(f"  🚫 BUY blocked: weekly signal bearish (RSI={w_rsi}, MACD={w_macd_hist})")
                print(f"  📋 {symbol}: daily=BUY but weekly=BEARISH — skipping")
                decision = "HOLD"
            elif portfolio.get("drawdown_halted"):
                print(f"  🚫 BUY blocked: drawdown circuit breaker active")
                decision = "HOLD"
            elif skip_buy:
                print(f"  🚫 BUY blocked: earnings blackout ({days_to_earnings} days)")
                decision = "HOLD"

        if decision in ("BUY", "SELL"):
            execute_trade(portfolio, decision, shares, symbol, data["price"], reasoning,
                          atr=data.get("atr"), rsi=data["rsi"], blended_score=blended_score,
                          spy_regime=spy_is_bull)
        else:
            print(f"  ⏸️  HOLD")
        log_to_obsidian(symbol, decision, shares, data["price"], confidence, reasoning,
                        data, signals, blended_score, adv_detail, adv_contrib)

    total = portfolio["cash"] + sum(h["shares"]*current_prices.get(s, h["avg_price"]) for s,h in portfolio["holdings"].items())

    # ── Drawdown circuit breaker: update peak and halted flag ─────────────────
    peak = portfolio.get("peak_value", STARTING_CASH)
    if total > peak:
        portfolio["peak_value"] = total
        peak = total
    dd_pct = (peak - total) / peak if peak > 0 else 0.0
    if dd_pct >= DRAWDOWN_CIRCUIT_BREAKER and not portfolio.get("drawdown_halted"):
        portfolio["drawdown_halted"] = True
        msg = f"🚨 CIRCUIT BREAKER: BUYs halted | DD: {dd_pct*100:.1f}% from peak ${peak:,.2f}"
        print(f"  {msg}")
        send_telegram(msg)
    elif portfolio.get("drawdown_halted") and dd_pct < DRAWDOWN_RESUME_PCT:
        portfolio["drawdown_halted"] = False
        msg = f"✅ Circuit breaker cleared: {dd_pct*100:.1f}% below peak"
        print(f"  {msg}")
        send_telegram(msg)

    pnl = total - STARTING_CASH
    perf = load_performance()
    if not perf.get("sp500_start") and sp500_price:
        perf["sp500_start"] = sp500_price
    perf["snapshots"].append({"date":datetime.now().strftime("%Y-%m-%d %H:%M"),"value":round(total,2),"cash":round(portfolio["cash"],2),"pnl":round(pnl,2)})
    save_performance(perf)
    save_portfolio(portfolio)
    try:
        save_signals(session_signals)
    except Exception as e:
        print(f"  [warn] Could not save signals.json: {e}")
    total_closed = perf.get("win_trades",0) + perf.get("loss_trades",0)
    wr = perf["win_trades"]/total_closed*100 if total_closed > 0 else 0
    summary = (f"📊 Session complete | Portfolio: ${total:,.2f} | "
               f"P&L: ${pnl:+,.2f} ({pnl/STARTING_CASH*100:+.2f}%) | "
               f"Win Rate: {wr:.1f}% | Regime: {regime_label}")
    print(f"\n{'='*55}")
    print(f"📈 Session Complete! Portfolio: ${total:,.2f} | P&L: ${pnl:+,.2f} ({pnl/STARTING_CASH*100:+.2f}%)")
    print(f"🏆 Win Rate: {wr:.1f}% | Cash: ${portfolio['cash']:,.2f}")
    print(f"{'='*55}\n")
    send_telegram(summary)

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
