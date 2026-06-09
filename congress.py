"""
congress.py - Congressional trading data for FridayTrader (FMP edition)
Source: Financial Modeling Prep /stable/house-trades + /stable/senate-trades
"""
import os
import time
import datetime as dt
from collections import defaultdict

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency. Run:  pip3 install requests")

FMP_KEY = os.environ.get("FMP_API_KEY", "")
HOUSE = "https://financialmodelingprep.com/stable/house-trades"
SENATE = "https://financialmodelingprep.com/stable/senate-trades"


def _normalize_action(txn_type):
    t = (txn_type or "").lower()
    if "purchase" in t or "buy" in t:
        return "BUY"
    if "sale" in t or "sell" in t:
        return "SELL"
    return "OTHER"


def _politician_name(row):
    for key in ("representative", "senator", "name"):
        if row.get(key):
            return row[key]
    first, last = row.get("firstName", ""), row.get("lastName", "")
    if first or last:
        return f"{first} {last}".strip()
    return row.get("office") or "Unknown"


def _fetch_one(url, symbol, chamber):
    out = []
    try:
        r = requests.get(url, params={"symbol": symbol, "apikey": FMP_KEY}, timeout=10)
        if r.status_code in (401, 403):
            print(f"[congress] {r.status_code} {chamber} {symbol} - key invalid or "
                  f"endpoint not in your plan.")
            return out
        r.raise_for_status()
        for row in r.json():
            out.append({
                "symbol": symbol,
                "chamber": chamber,
                "politician": _politician_name(row),
                "action": _normalize_action(row.get("type")),
                "raw_action": row.get("type", ""),
                "amount": row.get("amount", ""),
                "transaction_date": row.get("transactionDate", "") or row.get("date", ""),
                "filing_date": row.get("disclosureDate", ""),
            })
    except Exception as e:
        print(f"[congress] fetch failed {chamber} {symbol}: {e}")
    return out


def fetch_congress_trades(symbols, pause=0.15):
    if not FMP_KEY:
        print("No FMP_API_KEY set. Run: export FMP_API_KEY=\"your_key\"")
        return []
    trades = []
    for sym in symbols:
        trades += _fetch_one(HOUSE, sym, "House")
        trades += _fetch_one(SENATE, sym, "Senate")
        time.sleep(pause)
    return trades


def build_politician_profiles(trades):
    by_person = defaultdict(lambda: {"buys": 0, "sells": 0, "trades": []})
    for t in trades:
        p = by_person[t["politician"]]
        if t["action"] == "BUY":
            p["buys"] += 1
        elif t["action"] == "SELL":
            p["sells"] += 1
        p["trades"].append(t)
    profiles = []
    for name, p in by_person.items():
        p["trades"].sort(key=lambda x: x["transaction_date"], reverse=True)
        profiles.append({
            "politician": name,
            "total_trades": p["buys"] + p["sells"],
            "buys": p["buys"], "sells": p["sells"],
            "lean": ("Net Buyer" if p["buys"] > p["sells"]
                     else "Net Seller" if p["sells"] > p["buys"] else "Mixed"),
            "recent": p["trades"][:8],
        })
    profiles.sort(key=lambda x: x["total_trades"], reverse=True)
    return profiles


def congress_signal(trades, lookback_days=45):
    cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
    net = defaultdict(int)
    for t in trades:
        d = (t["transaction_date"] or "")[:10]
        if d and d >= cutoff:
            if t["action"] == "BUY":
                net[t["symbol"]] += 1
            elif t["action"] == "SELL":
                net[t["symbol"]] -= 1
    return {sym: max(-3, min(3, v)) for sym, v in net.items()}


if __name__ == "__main__":
    WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "NNE", "LEU", "CEG",
                 "CCJ", "RDW", "DCO", "COST", "TJX", "CAVA", "CMG", "VRT", "CRDO"]
    print("Fetching congressional trades for your watchlist...\n")
    tr = fetch_congress_trades(WATCHLIST)
    print(f"Got {len(tr)} trades.\n")
    profiles = build_politician_profiles(tr)
    if not profiles:
        print("No data. Check the key, or your watchlist had no recent congress trades.")
    else:
        print(f"{'POLITICIAN':<28}{'LEAN':<12}{'B/S':<10}TOTAL")
        print("-" * 58)
        for prof in profiles[:10]:
            bs = str(prof['buys']) + 'B/' + str(prof['sells']) + 'S'
            print(f"{prof['politician']:<28}{prof['lean']:<12}{bs:<10}{prof['total_trades']}")
    print("\nPer-symbol signal the bot would use:")
    print(congress_signal(tr))
