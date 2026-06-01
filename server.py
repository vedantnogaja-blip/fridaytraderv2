from flask import Flask, send_file, jsonify
import yfinance as yf
from datetime import datetime
import pytz
import json
import os

app = Flask(__name__)
VAULT = os.path.expanduser("~/Documents/FridayTrader")

@app.route("/")
def index():
    return send_file(os.path.join(VAULT, "dashboard.html"))

@app.route("/portfolio.json")
def portfolio():
    with open(os.path.join(VAULT, "portfolio.json")) as f:
        return jsonify(json.load(f))

@app.route("/prices")
def prices():
    ny = pytz.timezone("America/New_York")
    now = datetime.now(ny)
    minutes = now.hour * 60 + now.minute
    is_open = now.weekday() < 5 and 570 <= minutes < 960
    status = "Open" if is_open else "Closed"
    data = {}
    for sym in ["AAPL","NVDA","TSLA","MSFT","GOOGL"]:
        try:
            hist = yf.Ticker(sym).history(period="2d")
            if not hist.empty:
                cur = round(float(hist["Close"].iloc[-1]), 2)
                prev = round(float(hist["Close"].iloc[-2]), 2)
                data[sym] = {"price": cur, "change_pct": round((cur-prev)/prev*100,2), "prev_close": prev}
        except:
            pass
    return jsonify({"market_open": is_open, "market_status": status, "prices": data})


@app.route('/performance')
def performance():
    return send_file(os.path.join(VAULT, 'performance.html'))

@app.route('/performance.json')
def perf_json():
    import json
    path = os.path.join(VAULT, 'performance.json')
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({'snapshots':[],'win_trades':0,'loss_trades':0,'total_realized_pnl':0,'sp500_start':None})
if __name__ == "__main__":
    print("Server running at http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
