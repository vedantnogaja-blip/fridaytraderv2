"""
advanced_strategies.py - University-level strategy layer for FridayTrader
Three layers, all returning {symbol: int in [-3,+3]} to fold into your score:
  1) regime-aware (momentum in trends, mean-reversion in ranges)
  2) pairs / statistical arbitrage (cointegration)
  3) ML signal layer (gradient boosting, time-series CV)
Each function takes a price DataFrame: index=dates, columns=symbols, values=close.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
CLAMP = 3

def _clamp(x, lo=-CLAMP, hi=CLAMP):
    return int(max(lo, min(hi, round(x))))

def _frame(prices):
    df = prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)
    return df.sort_index().astype(float)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def efficiency_ratio(series, n=20):
    net = series.diff(n).abs()
    path = series.diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)

def detect_regime(series, n=20, trend_threshold=0.30):
    s = pd.Series(series).dropna()
    if len(s) < n + 1:
        return {"regime": "UNKNOWN", "direction": "FLAT", "er": None}
    er = float(efficiency_ratio(s, n).iloc[-1])
    ma = s.rolling(n).mean().iloc[-1]
    direction = "UP" if s.iloc[-1] >= ma else "DOWN"
    regime = "TREND" if er >= trend_threshold else "RANGE"
    return {"regime": regime, "direction": direction, "er": round(er, 3)}

def _momentum_score(series, lookback=63):
    s = pd.Series(series).dropna()
    if len(s) < lookback + 1:
        return 0
    ret = s.iloc[-1] / s.iloc[-lookback - 1] - 1.0
    return _clamp(ret / 0.05)

def _mean_reversion_score(series, n=20):
    s = pd.Series(series).dropna()
    if len(s) < n + 1:
        return 0
    ma = s.rolling(n).mean().iloc[-1]
    sd = s.rolling(n).std().iloc[-1]
    if not sd:
        return 0
    z = (s.iloc[-1] - ma) / sd
    return _clamp(-z)

def regime_aware_signal(prices, n=20, momentum_lookback=63):
    df = _frame(prices)
    scores, details = {}, {}
    for sym in df.columns:
        s = df[sym].dropna()
        reg = detect_regime(s, n=n)
        if reg["regime"] == "TREND":
            score, strat = _momentum_score(s, momentum_lookback), "momentum"
        elif reg["regime"] == "RANGE":
            score, strat = _mean_reversion_score(s, n), "mean_reversion"
        else:
            score, strat = 0, "none"
        scores[sym] = score
        details[sym] = {**reg, "strategy": strat, "score": score}
    return scores, details

from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def find_cointegrated_pairs(prices, p_threshold=0.05, min_obs=120):
    df = _frame(prices).dropna()
    if len(df) < min_obs:
        return []
    syms = list(df.columns)
    found = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = syms[i], syms[j]
            try:
                _, pval, _ = coint(df[a], df[b])
            except Exception:
                continue
            if pval < p_threshold:
                found.append({"a": a, "b": b, "pvalue": round(float(pval), 4)})
    return sorted(found, key=lambda x: x["pvalue"])

def _hedge_ratio(y, x):
    x_ = sm.add_constant(x)
    return float(sm.OLS(y, x_).fit().params.iloc[1])

def pairs_signal(prices, pairs, lookback=60, entry_z=2.0, exit_z=0.5):
    df = _frame(prices).dropna()
    scores, details = {}, {}
    for p in pairs:
        a, b = p["a"], p["b"]
        if a not in df or b not in df:
            continue
        window = df[[a, b]].tail(lookback)
        if len(window) < lookback:
            continue
        beta = _hedge_ratio(window[a], window[b])
        spread = window[a] - beta * window[b]
        z = (spread.iloc[-1] - spread.mean()) / (spread.std() or np.nan)
        if not np.isfinite(z):
            continue
        lean = 0
        if z > entry_z:
            scores[a] = scores.get(a, 0) - 2; scores[b] = scores.get(b, 0) + 2; lean = -1
        elif z < -entry_z:
            scores[a] = scores.get(a, 0) + 2; scores[b] = scores.get(b, 0) - 2; lean = 1
        details[f"{a}/{b}"] = {"beta": round(beta, 3), "z": round(float(z), 2),
                               "lean": lean, "pvalue": p.get("pvalue")}
    return {k: _clamp(v) for k, v in scores.items()}, details

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
FEATURE_COLS = ["ret_5", "ret_10", "ret_20", "rsi_14", "vol_20", "px_to_ma20", "macd_hist"]

def _features_for_symbol(series):
    s = pd.Series(series).astype(float)
    f = pd.DataFrame(index=s.index)
    f["ret_5"] = s.pct_change(5); f["ret_10"] = s.pct_change(10); f["ret_20"] = s.pct_change(20)
    f["rsi_14"] = rsi(s, 14); f["vol_20"] = s.pct_change().rolling(20).std()
    f["px_to_ma20"] = s / s.rolling(20).mean() - 1.0; f["macd_hist"] = macd_hist(s)
    return f

def build_dataset(prices, horizon=5):
    df = _frame(prices); rows = []
    for sym in df.columns:
        s = df[sym].dropna()
        if len(s) < 60:
            continue
        feats = _features_for_symbol(s)
        feats["label"] = (s.shift(-horizon) / s - 1.0 > 0).astype(int)
        feats["symbol"] = sym
        rows.append(feats)
    return pd.concat(rows).dropna()

def train_ml_model(prices, horizon=5, verbose=True):
    data = build_dataset(prices, horizon).sort_index()
    X, y = data[FEATURE_COLS], data["label"]
    if len(X) < 200 or y.nunique() < 2:
        if verbose: print("[ml] not enough data to train")
        return None, None
    aucs = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(X.iloc[tr], y.iloc[tr])
        try: aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
        except ValueError: pass
    cv_auc = float(np.mean(aucs)) if aucs else None
    final = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
    final.fit(X, y)
    if verbose and cv_auc: print(f"[ml] out-of-fold CV AUC = {cv_auc:.3f} (0.50 = coin flip)")
    return final, cv_auc

def ml_signal(model, prices):
    if model is None:
        return {}, {}
    df = _frame(prices); scores, details = {}, {}
    for sym in df.columns:
        feats = _features_for_symbol(df[sym].dropna()).dropna()
        if feats.empty:
            continue
        prob = float(model.predict_proba(feats[FEATURE_COLS].iloc[[-1]])[0, 1])
        scores[sym] = _clamp((prob - 0.5) * 6)
        details[sym] = {"prob_up": round(prob, 3), "score": scores[sym]}
    return scores, details

def combine_signals(prices, weights=None, ml_model=None, pairs=None):
    w = {"regime": 1.0, "pairs": 1.0, "ml": 1.0}
    if weights: w.update(weights)
    reg_scores, reg_detail = regime_aware_signal(prices)
    if pairs is None: pairs = find_cointegrated_pairs(prices)
    pair_scores, pair_detail = pairs_signal(prices, pairs)
    ml_scores, ml_detail = ml_signal(ml_model, prices)
    combined = {}
    for sym in _frame(prices).columns:
        combined[sym] = _clamp(w["regime"] * reg_scores.get(sym, 0)
                               + w["pairs"] * pair_scores.get(sym, 0)
                               + w["ml"] * ml_scores.get(sym, 0))
    return combined, {"regime": reg_detail, "pairs": pair_detail, "ml": ml_detail}

if __name__ == "__main__":
    rng = np.random.default_rng(7); days = 500
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    aaa = 100 + np.cumsum(rng.normal(0.05, 1.0, days))
    px = pd.DataFrame({"AAA": aaa, "BBB": 2.0*aaa + rng.normal(0,2.0,days) + 10,
                       "CCC": 50+np.cumsum(rng.normal(0,1,days)),
                       "DDD": 50+np.cumsum(rng.normal(0,1,days)),
                       "EEE": 50+np.cumsum(rng.normal(0,1,days))}, index=idx)
    print("regime:", regime_aware_signal(px)[0])
    pairs = find_cointegrated_pairs(px); print("pairs:", pairs)
    model, auc = train_ml_model(px)
    print("combined:", combine_signals(px, ml_model=model, pairs=pairs)[0])
