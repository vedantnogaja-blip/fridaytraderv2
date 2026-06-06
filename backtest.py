"""
backtest.py — Walk-forward backtest of FridayTrader v3 full scoring stack.

Split: first 70% in-sample (ML trained here only), last 30% out-of-sample.
Costs: 0.15% per side (0.1% commission + 0.05% slippage), applied on every fill.
Reports Sharpe, Sortino, max drawdown, total return vs SPY — separately per period.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

from config import WATCHLIST, ADVANCED_LAYER_WEIGHTS, ADVANCED_CAP
from advanced_strategies import (
    rsi as _rsi, macd_hist as _macd_hist,
    find_cointegrated_pairs, train_ml_model,
    _features_for_symbol, FEATURE_COLS, _clamp,
)

STARTING_CASH  = 10_000
TC             = 0.0015   # per-side transaction cost + slippage
STOP_LOSS      = 0.05
TAKE_PROFIT    = 0.15
MAX_POS_PCT    = 0.20
MAX_POSITIONS  = 5
MIN_BUY_SCORE  = 3
MAX_SELL_SCORE = -3
TRAIN_FRAC     = 0.70
REGIME_N       = 20
MOMENTUM_LB    = 63
PAIRS_LOOKBACK = 60


# ── Data ──────────────────────────────────────────────────────────────────────

def download_data():
    print("Downloading 2y of OHLCV data for watchlist + SPY...")
    tickers = WATCHLIST + ["SPY"]
    raw = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
    closes  = raw["Close"].dropna(how="all")
    volumes = raw["Volume"].reindex(closes.index).fillna(0)
    spy = closes["SPY"].dropna()
    closes  = closes.drop(columns=["SPY"], errors="ignore")
    volumes = volumes.drop(columns=["SPY"], errors="ignore")
    # drop symbols missing most data
    closes  = closes.loc[:, closes.notna().mean() > 0.8]
    volumes = volumes.reindex(columns=closes.columns)
    print(f"  {len(closes)} trading days, {len(closes.columns)} symbols")
    return closes, volumes, spy


# ── Technical score (vectorized over full history) ────────────────────────────

def compute_tech_scores(closes, volumes):
    scores = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for sym in closes.columns:
        px  = closes[sym].dropna()
        vol = volumes.get(sym, pd.Series(dtype=float)).reindex(px.index).fillna(0)

        r = _rsi(px, 14)
        s = pd.Series(0.0, index=r.index)
        s[r < 35] = 2.0; s[r > 70] = -2.0
        scores.loc[s.index, sym] += s

        mh = _macd_hist(px)
        scores.loc[mh.index, sym] += np.where(mh > 0, 2.0, -1.0)

        tr5 = px.pct_change(5) * 100
        s = pd.Series(0.0, index=tr5.index)
        s[tr5 > 2] = 2.0; s[tr5 < -2] = -2.0
        scores.loc[s.index, sym] += s

        avg_vol = vol.rolling(20).mean()
        vr = vol / avg_vol.replace(0, np.nan)
        s = pd.Series(0.0, index=vr.index)
        s[vr >= 1.5] = 1.0; s[vr < 0.8] = -1.0
        scores.loc[s.index, sym] += s

        hi = px.rolling(20).max(); lo = px.rolling(20).min()
        pp = (px - lo) / (hi - lo).replace(0, np.nan) * 100
        s = pd.Series(0.0, index=pp.index)
        s[pp < 30] = 1.0; s[pp > 80] = -1.0
        scores.loc[s.index, sym] += s

    return scores.clip(-8, 8)


# ── Regime score (vectorized, causal rolling windows) ────────────────────────

def compute_regime_scores(closes):
    scores = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for sym in closes.columns:
        px = closes[sym].dropna()
        if len(px) < REGIME_N + MOMENTUM_LB:
            continue

        net  = px.diff(REGIME_N).abs()
        path = px.diff().abs().rolling(REGIME_N).sum()
        er   = net / path.replace(0, np.nan)
        ma   = px.rolling(REGIME_N).mean()
        sd   = px.rolling(REGIME_N).std()

        is_trend = (er >= 0.30).reindex(px.index, fill_value=False)

        mom  = (px.pct_change(MOMENTUM_LB) / 0.05).clip(-3, 3).round()
        z    = (px - ma) / sd.replace(0, np.nan)
        mr   = (-z).clip(-3, 3).round()

        final = pd.Series(0.0, index=px.index)
        final[is_trend]  = mom[is_trend]
        final[~is_trend] = mr[~is_trend]
        scores.loc[final.index, sym] = final

    return scores.clip(-3, 3)


# ── Pairs score (uses hedge ratios fixed at in-sample fit) ────────────────────

def fit_pairs_betas(closes_train, pairs_raw):
    enriched = []
    for p in pairs_raw:
        a, b = p["a"], p["b"]
        if a not in closes_train.columns or b not in closes_train.columns:
            continue
        y = closes_train[a].dropna()
        x = closes_train[b].dropna()
        idx = y.index.intersection(x.index)
        if len(idx) < 30:
            continue
        x_ = sm.add_constant(x.loc[idx])
        beta = float(sm.OLS(y.loc[idx], x_).fit().params.iloc[1])
        enriched.append({**p, "beta": beta})
    return enriched


def compute_pairs_scores(closes, pairs):
    scores = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for p in pairs:
        a, b, beta = p["a"], p["b"], p.get("beta", 1.0)
        if a not in closes.columns or b not in closes.columns:
            continue
        spread    = closes[a] - beta * closes[b]
        roll_mean = spread.rolling(PAIRS_LOOKBACK).mean()
        roll_std  = spread.rolling(PAIRS_LOOKBACK).std()
        z = (spread - roll_mean) / roll_std.replace(0, np.nan)

        buy_a  = (z < -2.0).astype(float)
        sell_a = (z >  2.0).astype(float)
        scores[a] = (scores[a] + (buy_a - sell_a) * 2).clip(-3, 3)
        scores[b] = (scores[b] + (sell_a - buy_a) * 2).clip(-3, 3)

    return scores


# ── ML score (predict on all dates; model trained in-sample only) ─────────────

def compute_ml_scores(closes, model):
    scores = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    if model is None:
        return scores
    for sym in closes.columns:
        px    = closes[sym].dropna()
        feats = _features_for_symbol(px).dropna()
        if feats.empty:
            continue
        probs = model.predict_proba(feats[FEATURE_COLS])[:, 1]
        sym_s = pd.Series(np.clip(np.round((probs - 0.5) * 6), -3, 3), index=feats.index)
        scores.loc[sym_s.index, sym] = sym_s

    return scores


# ── Blend ─────────────────────────────────────────────────────────────────────

def blend(tech, regime, pairs, ml, weights=None, cap=ADVANCED_CAP):
    w = {"regime": 1.0, "pairs": 1.0, "ml": 1.0}
    if weights:
        w.update(weights)
    adv = (regime * w["regime"] + pairs * w["pairs"] + ml * w["ml"]).clip(-cap, cap)
    return tech.add(adv, fill_value=0)


# ── Trading simulation ────────────────────────────────────────────────────────

def simulate(dates, closes, scores):
    cash     = float(STARTING_CASH)
    holdings = {}   # sym -> {shares, avg_price}
    values   = []

    valid = [d for d in dates if d in closes.index and d in scores.index]

    for date in valid:
        px  = closes.loc[date]
        day = scores.loc[date]

        def safe_price(sym):
            p = px.get(sym)
            return float(p) if p is not None and not pd.isna(p) else None

        # ── exits ──────────────────────────────────────────────────────────
        to_sell = []
        for sym, h in list(holdings.items()):
            p = safe_price(sym)
            if p is None:
                continue
            pnl = (p - h["avg_price"]) / h["avg_price"]
            score = day.get(sym)
            if (pnl <= -STOP_LOSS or pnl >= TAKE_PROFIT or
                    (score is not None and not pd.isna(score) and score <= MAX_SELL_SCORE)):
                to_sell.append(sym)

        for sym in set(to_sell):
            if sym not in holdings:
                continue
            p = safe_price(sym)
            if p is None:
                continue
            cash += holdings.pop(sym)["shares"] * p * (1 - TC)

        # ── position sizing ───────────────────────────────────────────────
        total = cash + sum(
            h["shares"] * (safe_price(sym) or h["avg_price"])
            for sym, h in holdings.items()
        )

        # ── entries ───────────────────────────────────────────────────────
        slots = MAX_POSITIONS - len(holdings)
        if slots > 0:
            candidates = (
                day[day >= MIN_BUY_SCORE]
                .drop(index=list(holdings.keys()), errors="ignore")
                .sort_values(ascending=False)
            )
            for sym in candidates.index[:slots]:
                p = safe_price(sym)
                if p is None:
                    continue
                alloc  = min(total * MAX_POS_PCT, cash * 0.90)
                shares = int(alloc / (p * (1 + TC)))
                if shares <= 0:
                    continue
                cost = shares * p * (1 + TC)
                if cost > cash:
                    break
                cash -= cost
                holdings[sym] = {"shares": shares, "avg_price": p}

        port_val = cash + sum(
            h["shares"] * (safe_price(sym) or h["avg_price"])
            for sym, h in holdings.items()
        )
        values.append(port_val)

    return pd.Series(values, index=valid, dtype=float)


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(vals, label):
    rets = vals.pct_change().dropna()
    if len(rets) < 5 or rets.std() == 0:
        print(f"\n{label}: not enough data"); return {}

    sharpe  = rets.mean() / rets.std() * np.sqrt(252)
    neg     = rets[rets < 0]
    sortino = rets.mean() / neg.std() * np.sqrt(252) if len(neg) > 1 else float("inf")
    cum     = (1 + rets).cumprod()
    max_dd  = (1 - cum / cum.cummax()).max()
    total   = vals.iloc[-1] / vals.iloc[0] - 1

    w = 42
    print(f"\n{'─'*w}")
    print(f"  {label}")
    print(f"{'─'*w}")
    print(f"  Total Return : {total*100:+.1f}%")
    print(f"  Sharpe Ratio : {sharpe:.2f}")
    print(f"  Sortino Ratio: {sortino:.2f}")
    print(f"  Max Drawdown : {max_dd*100:.1f}%")
    return dict(total_return=total, sharpe=sharpe, sortino=sortino, max_drawdown=max_dd)


def spy_metrics(spy_series, label):
    rets = spy_series.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    total  = spy_series.iloc[-1] / spy_series.iloc[0] - 1
    print(f"  SPY {label}: {total*100:+.1f}%  (Sharpe {sharpe:.2f})")
    return total, sharpe


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    closes, volumes, spy = download_data()

    n          = len(closes)
    split_idx  = int(n * TRAIN_FRAC)
    train_idx  = closes.index[:split_idx]
    test_idx   = closes.index[split_idx:]
    split_date = closes.index[split_idx]

    print(f"\nIn-sample  : {closes.index[0].date()} → {train_idx[-1].date()} ({len(train_idx)} days)")
    print(f"Out-of-sample: {split_date.date()} → {closes.index[-1].date()} ({len(test_idx)} days)")

    train_closes = closes.loc[train_idx]

    # ── Train advanced components on in-sample data only ──────────────────
    print("\nFinding cointegrated pairs (in-sample only)...")
    raw_pairs   = find_cointegrated_pairs(train_closes)
    pairs       = fit_pairs_betas(train_closes, raw_pairs)
    pair_names  = [(p["a"], p["b"]) for p in pairs[:5]]
    print(f"  {len(pairs)} pairs found: {pair_names}")

    print("\nTraining ML model (in-sample only)...")
    ml_model, cv_auc = train_ml_model(train_closes, verbose=True)
    if cv_auc:
        print(f"  CV AUC = {cv_auc:.3f}  (0.50 = coin flip, >0.55 useful)")

    # ── Precompute all signal matrices over full history ───────────────────
    print("\nPrecomputing signal matrices...")
    tech   = compute_tech_scores(closes, volumes)
    regime = compute_regime_scores(closes)
    pair_s = compute_pairs_scores(closes, pairs)
    ml_s   = compute_ml_scores(closes, ml_model)
    blended = blend(tech, regime, pair_s, ml_s, weights=ADVANCED_LAYER_WEIGHTS)
    print("  Done.")

    # ── Simulations ───────────────────────────────────────────────────────
    print("\nRunning in-sample simulation...")
    is_vals  = simulate(train_idx, closes, blended)

    print("Running out-of-sample simulation...")
    oos_vals = simulate(test_idx, closes, blended)

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "="*42)
    print("  BACKTEST RESULTS")
    print("="*42)

    is_m  = metrics(is_vals,  f"IN-SAMPLE  ({closes.index[0].date()} – {train_idx[-1].date()})")
    oos_m = metrics(oos_vals, f"OUT-OF-SAMPLE ({split_date.date()} – {closes.index[-1].date()})")

    spy_is  = spy.reindex(train_idx).dropna()
    spy_oos = spy.reindex(test_idx).dropna()

    print(f"\n{'─'*42}")
    print(f"  SPY Benchmark")
    print(f"{'─'*42}")
    spy_is_ret,  spy_is_sh  = spy_metrics(spy_is,  "in-sample")
    spy_oos_ret, spy_oos_sh = spy_metrics(spy_oos, "out-of-sample")

    print(f"\n{'─'*42}")
    print(f"  Alpha vs SPY")
    print(f"{'─'*42}")
    print(f"  In-sample  alpha : {is_m.get('total_return',0)*100 - spy_is_ret*100:+.1f}%")
    print(f"  Out-of-sample α  : {oos_m.get('total_return',0)*100 - spy_oos_ret*100:+.1f}%")

    if oos_m.get("sharpe", 0) < 0.5:
        print("\n  ⚠  Out-of-sample Sharpe < 0.5 — strategy does not robustly")
        print("     generalise beyond the training period. This is a real finding.")
    if oos_m.get("total_return", 0) < spy_oos_ret:
        print("  ⚠  Out-of-sample return trails SPY — no positive alpha detected.")

    # ── quantstats tearsheet (optional) ───────────────────────────────────
    try:
        import quantstats as qs
        full_vals = pd.concat([is_vals, oos_vals]).sort_index()
        full_rets = full_vals.pct_change().dropna()
        spy_bench = spy.reindex(full_rets.index).pct_change().dropna().reindex(full_rets.index).fillna(0)
        qs.reports.html(full_rets, benchmark=spy_bench,
                        output="backtest_report.html",
                        title="FridayTrader v3 Backtest")
        print("\nquantstats HTML report → backtest_report.html")
    except ImportError:
        print("\n(quantstats not installed — pip install quantstats for HTML tearsheet)")
    except Exception as e:
        print(f"\n(quantstats: {e})")
