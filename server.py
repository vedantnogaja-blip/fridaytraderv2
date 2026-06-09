from flask import Flask, send_file, jsonify
import yfinance as yf
from datetime import datetime
import pytz
import json
import os
import threading
import time

from friday_trader_v3 import WATCHLIST  # single source of truth (17 symbols)

app = Flask(__name__)

VAULT = os.path.expanduser("~/Documents/FridayTrader")
NY = pytz.timezone("America/New_York")

# --- Background price poller -------------------------------------------------
# A daemon thread fetches every WATCHLIST symbol in two batched yfinance calls
# (one intraday for the latest price, one daily for prev close) on a timer, and
# stores the result in _price_data behind a lock. The /prices route ONLY reads
# this dict, so it always responds instantly and never triggers a live fetch.
_price_data = {}
_price_lock = threading.Lock()
POLL_OPEN_SECONDS = 15      # market open: refresh quickly
POLL_CLOSED_SECONDS = 60    # market closed: refresh lazily


def market_open(now=None):
    now = now or datetime.now(NY)
    minutes = now.hour * 60 + now.minute
    return now.weekday() < 5 and 570 <= minutes < 960  # 09:30–16:00 ET, Mon–Fri


def _fetch_prices():
    """Fetch all WATCHLIST prices in two batched calls. Returns {sym: {...}}.

    Raises if the batched download itself fails; individual symbols that can't
    be parsed are simply skipped (so one bad ticker can't sink the whole set).
    """
    intraday = yf.download(
        tickers=WATCHLIST, period="2d", interval="1m",
        group_by="ticker", threads=True, progress=False,
    )
    daily = yf.download(
        tickers=WATCHLIST, period="2d", interval="1d",
        group_by="ticker", threads=True, progress=False,
    )
    result = {}
    for sym in WATCHLIST:
        try:
            closes = intraday[sym]["Close"].dropna()            # latest traded price
            day_closes = daily[sym]["Close"].dropna()           # prior trading day close
            if closes.empty or day_closes.empty:
                continue
            cur = round(float(closes.iloc[-1]), 2)
            prev = round(float(day_closes.iloc[-2] if len(day_closes) >= 2 else day_closes.iloc[-1]), 2)
            change_pct = round((cur - prev) / prev * 100, 2) if prev else 0.0
            result[sym] = {"price": cur, "change_pct": change_pct, "prev_close": prev}
        except Exception:
            continue
    return result


def _poll_loop():
    while True:
        try:
            fresh = _fetch_prices()
            if fresh:
                # Per-symbol merge: a symbol missing from this poll keeps its
                # last-good value rather than disappearing.
                with _price_lock:
                    _price_data.update(fresh)
        except Exception as e:
            # Transient failure — log it and leave the previous values in place.
            print(f"[poller] price fetch failed, serving last-good prices: {e}")
        time.sleep(POLL_OPEN_SECONDS if market_open() else POLL_CLOSED_SECONDS)


def start_poller():
    """Warm the cache once synchronously, then start the background poller."""
    try:
        warm = _fetch_prices()
        if warm:
            with _price_lock:
                _price_data.update(warm)
    except Exception as e:
        print(f"[startup] initial price fetch failed: {e}")
    threading.Thread(target=_poll_loop, daemon=True).start()


@app.route("/")
def index():
    return send_file(os.path.join(VAULT, "dashboard.html"))


@app.route("/portfolio.json")
def portfolio():
    with open(os.path.join(VAULT, "portfolio.json")) as f:
        return jsonify(json.load(f))


@app.route("/prices")
def prices():
    is_open = market_open()
    with _price_lock:
        data = dict(_price_data)  # snapshot; never fetch on the request path
    return jsonify({
        "market_open": is_open,
        "market_status": "Open" if is_open else "Closed",
        "prices": data,
    })


@app.route('/performance')
def performance():
    return send_file(os.path.join(VAULT, 'performance.html'))


@app.route('/performance.json')
def perf_json():
    path = os.path.join(VAULT, 'performance.json')
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({'snapshots': [], 'win_trades': 0, 'loss_trades': 0, 'total_realized_pnl': 0, 'sp500_start': None})


@app.route('/indices')
def indices():
    result = {}
    for key, symbol in [('sp500', '^GSPC'), ('nasdaq', '^IXIC'), ('djia', '^DJI')]:
        try:
            h = yf.Ticker(symbol).history(period='2d')
            if not h.empty and len(h) >= 2:
                cur = round(float(h['Close'].iloc[-1]), 2)
                prev = round(float(h['Close'].iloc[-2]), 2)
                result[key] = {'price': cur, 'prev_close': prev, 'change': round(cur - prev, 2), 'change_pct': round((cur - prev) / prev * 100, 2)}
        except Exception:
            pass
    return jsonify(result)


@app.route("/health")
def health():
    port_path = os.path.join(VAULT, "portfolio.json")
    perf_path = os.path.join(VAULT, "performance.json")
    port = {}
    perf = {}
    try:
        if os.path.exists(port_path):
            with open(port_path) as f:
                port = json.load(f)
    except Exception: pass
    try:
        if os.path.exists(perf_path):
            with open(perf_path) as f:
                perf = json.load(f)
    except Exception: pass
    holdings = port.get("holdings", {})
    portfolio_value = round(port.get("cash", 0) + sum(
        h["shares"] * h["avg_price"] for h in holdings.values()
    ), 2)
    snaps = perf.get("snapshots", [])
    last_run = snaps[-1]["date"] if snaps else None
    return jsonify({
        "last_run": last_run,
        "next_run": "Weekdays 22:00 + 02:00 SGT",
        "circuit_breaker_active": port.get("drawdown_halted", False),
        "portfolio_value": portfolio_value,
        "holdings_count": len(holdings),
        "sessions_run": port.get("sessions", 0),
        "errors_last_run": []
    })


@app.route("/signals")
def signals():
    path = os.path.join(VAULT, "signals.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({"timestamp": None, "signals": []})


@app.route("/backtest-report")
def backtest_report():
    report_dir = os.path.join(VAULT, "backtest_results")
    if not os.path.exists(report_dir):
        return jsonify({"error": "No backtest results found. Run python3 backtest.py first."})
    json_files = sorted(
        [f for f in os.listdir(report_dir) if f.endswith(".json")],
        reverse=True
    )
    if not json_files:
        return jsonify({"error": "No backtest JSON found. Run python3 backtest.py first."})
    try:
        with open(os.path.join(report_dir, json_files[0])) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    start_poller()
    print("Server running at http://localhost:9090")
    app.run(host="0.0.0.0", port=9090, debug=False)
