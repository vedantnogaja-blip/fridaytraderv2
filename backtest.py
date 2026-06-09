"""
backtest.py — Walk-forward backtest of FridayTrader v3 full scoring stack.

Split: first 70% in-sample (ML trained here only), last 30% out-of-sample.
Costs: ROUND_TRIP_COST = 0.003 applied symmetrically per fill (0.30% on buy,
       0.30% on sell) → ~0.60% effective round-trip.  Realistic for a mixed
       large/small-cap watchlist with market-impact slippage on thinner names.
Fill: signal computed at day-T close; order executes at day-(T+1) open.
      One-bar execution lag — same-day close fills are not achievable in practice.
Reports Sharpe, Sortino, max drawdown, total return vs SPY — separately per period.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

from config import (WATCHLIST, ADVANCED_LAYER_WEIGHTS, ADVANCED_CAP,
                    ATR_STOP_MULT, ATR_TARGET_MULT,
                    DRAWDOWN_CIRCUIT_BREAKER, DRAWDOWN_RESUME_PCT)
from advanced_strategies import (
    rsi as _rsi, macd_hist as _macd_hist,
    find_cointegrated_pairs, train_ml_model,
    _features_for_symbol, FEATURE_COLS, _clamp,
)

STARTING_CASH   = 10_000
ROUND_TRIP_COST = 0.003   # 0.10% commission + 0.20% slippage per fill; ~0.60% round-trip
STOP_LOSS       = 0.05
TAKE_PROFIT     = 0.15
MAX_POS_PCT     = 0.20
MAX_POSITIONS   = 5
MIN_BUY_SCORE   = 3
MAX_SELL_SCORE  = -3
TRAIN_FRAC      = 0.70
REGIME_N        = 20
MOMENTUM_LB     = 63
PAIRS_LOOKBACK  = 60


# ── Data ──────────────────────────────────────────────────────────────────────

def download_data():
    print("Downloading 2y of OHLCV data for watchlist + SPY...")
    tickers = WATCHLIST + ["SPY"]
    raw = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
    closes  = raw["Close"].dropna(how="all")
    opens   = raw["Open"].reindex(closes.index)
    highs   = raw["High"].reindex(closes.index)
    lows    = raw["Low"].reindex(closes.index)
    volumes = raw["Volume"].reindex(closes.index).fillna(0)
    spy = closes["SPY"].dropna()
    for df in [closes, opens, highs, lows, volumes]:
        for col in ["SPY"]:
            if col in df.columns:
                df.drop(columns=col, inplace=True, errors="ignore")
    # drop symbols missing most data
    closes  = closes.loc[:, closes.notna().mean() > 0.8]
    opens   = opens.reindex(columns=closes.columns)
    highs   = highs.reindex(columns=closes.columns)
    lows    = lows.reindex(columns=closes.columns)
    volumes = volumes.reindex(columns=closes.columns)
    print(f"  {len(closes)} trading days, {len(closes.columns)} symbols")
    return closes, opens, highs, lows, volumes, spy


# ── ATR matrix (Wilder's 14-period) ──────────────────────────────────────────

def compute_atr_matrix(closes, highs, lows, period=14):
    """Return a DataFrame of ATR-14 values, same shape as closes."""
    atrs = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for sym in closes.columns:
        h = highs[sym].reindex(closes.index).ffill()
        l = lows[sym].reindex(closes.index).ffill()
        c = closes[sym].reindex(closes.index).ffill()
        c_prev = c.shift(1)
        tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
        # Wilder's smoothing: alpha = 1/period
        atrs[sym] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atrs


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
# Fill model: signal observed at day-T close → order fills at day-(T+1) open.
# Stop/target: ATR-based (entry_price ± ATR_14 * multiplier from config).
# Drawdown circuit breaker: halts new BUYs when portfolio falls >8% from peak.

def simulate(dates, closes, opens, scores, atrs):
    cash         = float(STARTING_CASH)
    holdings     = {}   # sym -> {shares, avg_price, entry_atr}
    values       = []
    pending      = {}   # sym -> 'buy'|'sell'
    pending_atrs = {}   # sym -> ATR recorded at queue time (BUY orders only)
    peak_value   = float(STARTING_CASH)
    buy_halted   = False
    closed_trades = []

    valid = [d for d in dates if d in closes.index and d in scores.index]

    def _px(prices, sym):
        p = prices.get(sym)
        return float(p) if p is not None and not pd.isna(p) else None

    def _atr(date, sym):
        try:
            v = atrs.loc[date, sym]
            return float(v) if not pd.isna(v) else 0.0
        except (KeyError, TypeError):
            return 0.0

    for i, date in enumerate(valid):
        close_px = closes.loc[date]
        day      = scores.loc[date] if date in scores.index else pd.Series(dtype=float)
        open_px  = opens.loc[date] if date in opens.index else close_px

        # ── Step 1: execute pending orders at T's OPEN ────────────────────
        for sym, action in list(pending.items()):
            if action == 'sell' and sym in holdings:
                p = _px(open_px, sym) or _px(close_px, sym)
                if p:
                    h = holdings[sym]   # read BEFORE pop
                    entry_p = h["avg_price"]
                    entry_atr_v = h.get("entry_atr", 0.0)
                    # approximate net P&L including round-trip costs
                    net_pnl_pct = (p / entry_p - 1.0) - 2 * ROUND_TRIP_COST
                    closed_trades.append({
                        "sym": sym,
                        "entry_price": round(entry_p, 4),
                        "exit_price": round(p, 4),
                        "shares": h["shares"],
                        "pnl_pct": round(net_pnl_pct, 6),
                        "pnl_dollar": round(h["shares"] * entry_p * net_pnl_pct, 2),
                    })
                    cash += holdings.pop(sym)["shares"] * p * (1 - ROUND_TRIP_COST)

        total = cash + sum(
            h["shares"] * (_px(close_px, s) or h["avg_price"])
            for s, h in holdings.items()
        )
        for sym, action in sorted(pending.items(),
                                  key=lambda kv: day.get(kv[0], 0), reverse=True):
            if action == 'buy' and sym not in holdings:
                p = _px(open_px, sym) or _px(close_px, sym)
                if not p:
                    continue
                alloc  = min(total * MAX_POS_PCT, cash * 0.90)
                shares = int(alloc / (p * (1 + ROUND_TRIP_COST)))
                if shares <= 0:
                    continue
                cost = shares * p * (1 + ROUND_TRIP_COST)
                if cost > cash:
                    continue
                cash -= cost
                entry_atr = pending_atrs.pop(sym, 0.0)
                holdings[sym] = {"shares": shares, "avg_price": p, "entry_atr": entry_atr}
        pending      = {}
        pending_atrs = {}

        # ── Step 2: check stops + score exits at T's CLOSE ───────────────
        for sym, h in list(holdings.items()):
            p = _px(close_px, sym)
            if p is None:
                continue
            entry_atr = h.get("entry_atr", 0.0)
            if entry_atr > 0:
                sell = (p <= h["avg_price"] - entry_atr * ATR_STOP_MULT or
                        p >= h["avg_price"] + entry_atr * ATR_TARGET_MULT)
            else:
                pnl  = (p - h["avg_price"]) / h["avg_price"]
                sell = pnl <= -STOP_LOSS or pnl >= TAKE_PROFIT
            score = day.get(sym)
            sell  = sell or (score is not None and not pd.isna(score) and score <= MAX_SELL_SCORE)
            if sell:
                pending[sym] = 'sell'

        # ── Step 3: record portfolio value + update drawdown breaker ──────
        port_val = cash + sum(
            h["shares"] * (_px(close_px, s) or h["avg_price"])
            for s, h in holdings.items()
        )
        values.append(port_val)

        if port_val > peak_value:
            peak_value = port_val
        dd = (peak_value - port_val) / peak_value if peak_value > 0 else 0.0
        if dd >= DRAWDOWN_CIRCUIT_BREAKER:
            buy_halted = True
        elif buy_halted and dd < DRAWDOWN_RESUME_PCT:
            buy_halted = False

        # ── Step 4: queue new buys for T+1 open ──────────────────────────
        occupied = set(holdings) | {s for s, a in pending.items() if a == 'sell'}
        slots    = MAX_POSITIONS - (len(holdings) - len([s for s in pending if s in holdings]))
        if slots > 0 and not buy_halted:
            candidates = (
                day[day >= MIN_BUY_SCORE]
                .drop(index=list(occupied), errors="ignore")
                .sort_values(ascending=False)
            )
            for sym in candidates.index[:slots]:
                pending[sym] = 'buy'
                pending_atrs[sym] = _atr(date, sym)

    return pd.Series(values, index=valid, dtype=float), closed_trades


# ── Metrics (kept for backward compat) ────────────────────────────────────────

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


# ── Full metrics (production standard) ───────────────────────────────────────

def full_metrics(vals, closed_trades, spy_series=None, label=""):
    if len(vals) < 5:
        return {}
    rets = vals.pct_change().dropna()
    if rets.std() == 0:
        return {}

    # Core metrics
    sharpe   = rets.mean() / rets.std() * np.sqrt(252)
    neg      = rets[rets < 0]
    sortino  = rets.mean() / neg.std() * np.sqrt(252) if len(neg) > 1 else float("inf")
    cum      = (1 + rets).cumprod()
    max_dd   = float((1 - cum / cum.cummax()).max())
    total    = float(vals.iloc[-1] / vals.iloc[0] - 1)
    n_days   = len(vals)
    ann_ret  = float((1 + total) ** (252 / n_days) - 1)
    calmar   = ann_ret / max_dd if max_dd > 0 else float("inf")

    # Max drawdown duration (consecutive bars in drawdown)
    dd_s = (1 - cum / cum.cummax())
    max_dd_dur = cur = 0
    for v in dd_s > 0:
        cur = cur + 1 if v else 0
        max_dd_dur = max(max_dd_dur, cur)

    # Trade-level stats
    n_trades = len(closed_trades)
    if n_trades > 0:
        wins   = [t for t in closed_trades if t["pnl_dollar"] > 0]
        losses = [t for t in closed_trades if t["pnl_dollar"] <= 0]
        win_rate   = len(wins) / n_trades
        avg_win    = float(np.mean([t["pnl_dollar"] for t in wins]))    if wins   else 0.0
        avg_loss   = float(abs(np.mean([t["pnl_dollar"] for t in losses]))) if losses else 0.0
        avg_wl     = avg_win / avg_loss if avg_loss > 0 else float("inf")
        g_profit   = sum(t["pnl_dollar"] for t in wins)
        g_loss     = abs(sum(t["pnl_dollar"] for t in losses))
        profit_fac = g_profit / g_loss if g_loss > 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = avg_wl = profit_fac = None

    # Alpha / Beta
    alpha = beta = None
    if spy_series is not None:
        spy_r = spy_series.pct_change().dropna()
        idx   = rets.index.intersection(spy_r.index)
        if len(idx) > 10:
            pr, sr = rets.loc[idx], spy_r.loc[idx]
            var_s  = float(np.var(sr))
            beta   = float(np.cov(pr, sr)[0, 1] / var_s) if var_s > 0 else 0.0
            alpha  = float((pr.mean() - beta * sr.mean()) * 252)

    return {
        "label": label,
        "total_return":         round(total,   4),
        "ann_return":           round(ann_ret, 4),
        "sharpe":               round(float(sharpe),  3),
        "sortino":              round(float(sortino), 3),
        "calmar":               round(calmar, 3) if calmar != float("inf") else None,
        "max_drawdown":         round(max_dd,  4),
        "max_dd_duration_days": max_dd_dur,
        "n_trades":             n_trades,
        "win_rate":             round(win_rate, 4) if win_rate is not None else None,
        "avg_win_dollar":       round(avg_win,  2)  if avg_win  is not None else None,
        "avg_loss_dollar":      round(avg_loss, 2)  if avg_loss is not None else None,
        "avg_win_loss_ratio":   round(avg_wl, 3)   if avg_wl   is not None and avg_wl != float("inf") else None,
        "profit_factor":        round(profit_fac, 3) if profit_fac is not None and profit_fac != float("inf") else None,
        "alpha":                round(alpha, 4) if alpha is not None else None,
        "beta":                 round(beta,  3) if beta  is not None else None,
    }


# ── Walk-forward layer ablation ──────────────────────────────────────────────

def walk_forward_layer_analysis(closes, opens, atrs, tech, regime, pair_s, ml_s, test_idx, split_date):
    """Run OOS simulation with each advanced layer isolated to measure standalone contribution."""
    z = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    configs = {
        "tech_only   ": blend(tech, z, z, z),
        "regime_only ": blend(tech, regime, z, z),
        "pairs_only  ": blend(tech, z, pair_s, z),
        "ml_only     ": blend(tech, z, z, ml_s),
        "full_stack  ": blend(tech, regime, pair_s, ml_s),
    }

    w = 48
    print(f"\n{'='*w}")
    print(f"  WALK-FORWARD LAYER ABLATION  (OOS: {split_date.date()} onward)")
    print(f"{'='*w}")
    print(f"  {'Config':<14}  {'OOS Return':>10}  {'OOS Sharpe':>10}  {'Max DD':>8}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*8}")

    results = {}
    for name, blended in configs.items():
        vals, _ = simulate(test_idx, closes, opens, blended, atrs)
        if len(vals) < 5:
            results[name] = {}
            continue
        rets   = vals.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        total  = vals.iloc[-1] / vals.iloc[0] - 1
        cum    = (1 + rets).cumprod()
        maxdd  = (1 - cum / cum.cummax()).max()
        results[name] = {"sharpe": sharpe, "total_return": total, "max_drawdown": maxdd}
        print(f"  {name}  {total*100:>+9.1f}%  {sharpe:>10.2f}  {maxdd*100:>7.1f}%")

    baseline_sh = results.get("tech_only   ", {}).get("sharpe", 0)
    full_sh     = results.get("full_stack  ", {}).get("sharpe", 0)

    print(f"\n  {'─'*w}")
    print(f"  RECOMMENDATION (honesty first)")
    print(f"  {'─'*w}")
    for layer, key in [("regime", "regime_only "), ("pairs", "pairs_only  "), ("ml", "ml_only     ")]:
        sh    = results.get(key, {}).get("sharpe", 0)
        delta = sh - baseline_sh
        if delta > 0.10:
            verdict = "KEEP  ✓  adds meaningful edge"
        elif delta > 0.0:
            verdict = "WEAK  ~  marginal improvement, monitor"
        else:
            verdict = "DROP  ✗  hurts or neutral vs tech-only"
        print(f"  {layer:8s}: Sharpe {sh:+.2f}  (Δ vs tech-only {delta:+.2f})  → {verdict}")

    print(f"\n  Full stack Sharpe {full_sh:.2f} vs tech-only {baseline_sh:.2f}")
    if full_sh < baseline_sh - 0.05:
        print("  ⚠  All three layers combined are net noise — consider dropping all advanced signals.")
    elif full_sh < baseline_sh:
        print("  ⚠  Full stack slightly trails tech-only — advanced layers are not helping.")
    else:
        print("  ✓  Full stack beats tech-only; keep layers with positive individual delta.")
    return results


# ── Rolling walk-forward ──────────────────────────────────────────────────────

def walk_forward(closes, opens, highs, lows, volumes, spy, atrs,
                 train_days=126, test_days=21):
    n = len(closes)
    windows = []
    start = 0
    while start + train_days + test_days <= n:
        windows.append((start, start + train_days, start + train_days + test_days))
        start += test_days

    print(f"\nWalk-forward: {len(windows)} windows "
          f"({train_days}-day train, {test_days}-day OOS each)")

    results    = []
    all_vals   = []
    all_trades = []

    for i, (t0, t1, t2) in enumerate(windows):
        train_closes = closes.iloc[t0:t1]
        test_idx     = closes.index[t1:t2]
        print(f"  Window {i+1:02d}/{len(windows)}: "
              f"train {closes.index[t0].date()} – {closes.index[t1-1].date()} | "
              f"test  {closes.index[t1].date()} – {closes.index[t2-1].date()}", end="", flush=True)

        # Per-window training (fast: 50 estimators, 3 splits)
        pairs_raw = find_cointegrated_pairs(train_closes)
        pairs_fit = fit_pairs_betas(train_closes, pairs_raw)
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from advanced_strategies import build_dataset, FEATURE_COLS
            data = build_dataset(train_closes).sort_index()
            X, y = data[FEATURE_COLS], data["label"]
            ml_model = None
            if len(X) >= 200 and y.nunique() >= 2:
                ml_model = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
                ml_model.fit(X, y)
        except Exception:
            ml_model = None

        # Causal signal computation over data available at test time
        avail_closes  = closes.iloc[:t2]
        avail_volumes = volumes.iloc[:t2]
        avail_opens   = opens.iloc[:t2]
        avail_highs   = highs.iloc[:t2]
        avail_lows    = lows.iloc[:t2]

        tech    = compute_tech_scores(avail_closes, avail_volumes)
        regime  = compute_regime_scores(avail_closes)
        pair_s  = compute_pairs_scores(avail_closes, pairs_fit)
        ml_s    = compute_ml_scores(avail_closes, ml_model)
        blended = blend(tech, regime, pair_s, ml_s, weights=ADVANCED_LAYER_WEIGHTS)
        w_atrs  = compute_atr_matrix(avail_closes, avail_highs, avail_lows)

        vals, trades = simulate(test_idx, avail_closes, avail_opens, blended, w_atrs)
        spy_w = spy.reindex(test_idx).dropna()
        m = full_metrics(vals, trades, spy_w, label=f"Window {i+1}")
        print(f"  → return={m.get('total_return', 0)*100:+.1f}%  Sharpe={m.get('sharpe', 0):.2f}")

        results.append({
            "window":      i + 1,
            "train_start": str(closes.index[t0].date()),
            "train_end":   str(closes.index[t1 - 1].date()),
            "test_start":  str(closes.index[t1].date()),
            "test_end":    str(closes.index[t2 - 1].date()),
            "n_pairs":     len(pairs_fit),
            "ml_trained":  ml_model is not None,
            **m
        })
        all_vals.append(vals)
        all_trades.extend(trades)

    # Aggregate OOS series
    combined_vals = pd.concat(all_vals).sort_index() if all_vals else pd.Series(dtype=float)
    return results, combined_vals, all_trades


# ── Monte Carlo random baseline ───────────────────────────────────────────────

def monte_carlo_baseline(closes, opens, atrs, test_idx, n_runs=100, seed=42):
    print(f"\nMonte Carlo random baseline ({n_runs} runs)...", end="", flush=True)
    rng = np.random.default_rng(seed)
    sharpes, returns = [], []
    for _ in range(n_runs):
        rand_s = pd.DataFrame(
            rng.uniform(-8, 8, size=(len(closes), len(closes.columns))),
            index=closes.index, columns=closes.columns
        )
        vals, _ = simulate(test_idx, closes, opens, rand_s, atrs)
        if len(vals) < 5:
            continue
        rets = vals.pct_change().dropna()
        if rets.std() > 0:
            sharpes.append(float(rets.mean() / rets.std() * np.sqrt(252)))
            returns.append(float(vals.iloc[-1] / vals.iloc[0] - 1))
    print(f" done ({len(sharpes)} valid)")
    if not sharpes:
        return {}
    return {
        "n_runs":      n_runs,
        "sharpe_p10":  round(float(np.percentile(sharpes, 10)), 3),
        "sharpe_p50":  round(float(np.percentile(sharpes, 50)), 3),
        "sharpe_p90":  round(float(np.percentile(sharpes, 90)), 3),
        "return_p10":  round(float(np.percentile(returns, 10)), 4),
        "return_p50":  round(float(np.percentile(returns, 50)), 4),
        "return_p90":  round(float(np.percentile(returns, 90)), 4),
    }


# ── Benchmarks ────────────────────────────────────────────────────────────────

def benchmark_spy_bh(spy, test_idx):
    s = spy.reindex(test_idx).dropna()
    if len(s) < 5:
        return {}
    rets = s.pct_change().dropna()
    if rets.std() == 0:
        return {}
    total  = float(s.iloc[-1] / s.iloc[0] - 1)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
    neg    = rets[rets < 0]
    sortino = float(rets.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 else float("inf")
    cum    = (1 + rets).cumprod()
    max_dd = float((1 - cum / cum.cummax()).max())
    return {
        "total_return": round(total, 4),
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_drawdown": round(max_dd, 4),
    }


def benchmark_60_40(spy, test_idx):
    try:
        agg_raw = yf.download(["AGG"], period="2y", auto_adjust=True, progress=False)
        if isinstance(agg_raw.columns, pd.MultiIndex):
            agg = agg_raw["Close"]["AGG"].dropna()
        else:
            agg = agg_raw["Close"].squeeze().dropna()
    except Exception:
        return {}
    spy_t = spy.reindex(test_idx).dropna()
    agg_t = agg.reindex(test_idx).dropna()
    idx   = spy_t.index.intersection(agg_t.index)
    if len(idx) < 5:
        return {}
    sr  = spy_t.loc[idx].pct_change().dropna()
    ar  = agg_t.loc[idx].pct_change().dropna()
    idx2 = sr.index.intersection(ar.index)
    blended = 0.6 * sr.loc[idx2] + 0.4 * ar.loc[idx2]
    if blended.std() == 0:
        return {}
    total   = float((1 + blended).cumprod().iloc[-1] - 1)
    sharpe  = float(blended.mean() / blended.std() * np.sqrt(252))
    neg     = blended[blended < 0]
    sortino = float(blended.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 else float("inf")
    cum    = (1 + blended).cumprod()
    max_dd = float((1 - cum / cum.cummax()).max())
    return {
        "total_return": round(total, 4),
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_drawdown": round(max_dd, 4),
    }


# ── Pretty-print metrics ──────────────────────────────────────────────────────

def print_full_metrics(m, label):
    if not m:
        print(f"\n  {label}: no data")
        return
    w = 50
    print(f"\n{'─'*w}")
    print(f"  {label}")
    print(f"{'─'*w}")
    print(f"  {'Total Return':<22}: {m.get('total_return', 0)*100:>+8.2f}%"
          f"   Ann: {m.get('ann_return', 0)*100:>+8.2f}%")
    print(f"  {'Sharpe':<22}: {m.get('sharpe', 0):>8.3f}"
          f"   Sortino: {m.get('sortino', 0):>8.3f}")
    calmar_str = f"{m['calmar']:.3f}" if m.get('calmar') is not None else "N/A"
    print(f"  {'Calmar':<22}: {calmar_str:>8}")
    print(f"  {'Max Drawdown':<22}: {m.get('max_drawdown', 0)*100:>8.2f}%"
          f"   (dur: {m.get('max_dd_duration_days', 0)} days)")
    print(f"  {'# Trades':<22}: {m.get('n_trades', 0):>8}")
    wr = m.get('win_rate')
    print(f"  {'Win Rate':<22}: {(wr*100 if wr is not None else 0):>8.1f}%")
    pf = m.get('profit_factor')
    pf_str = f"{pf:.3f}" if pf is not None else "N/A"
    print(f"  {'Profit Factor':<22}: {pf_str:>8}")
    wl = m.get('avg_win_loss_ratio')
    wl_str = f"{wl:.3f}" if wl is not None else "N/A"
    print(f"  {'Avg Win/Loss Ratio':<22}: {wl_str:>8}")
    alpha_str = f"{m['alpha']*100:+.2f}%" if m.get('alpha') is not None else "N/A"
    beta_str  = f"{m['beta']:.3f}"         if m.get('beta')  is not None else "N/A"
    print(f"  {'Alpha vs SPY':<22}: {alpha_str:>8}   Beta: {beta_str}")


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(window_results, agg_m, spy_bh, balanced, mc_baseline, closes, combined_vals, spy):
    base_dir = os.path.expanduser("~/Documents/FridayTrader")
    out_dir  = os.path.join(base_dir, "backtest_results")
    os.makedirs(out_dir, exist_ok=True)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path  = os.path.join(out_dir, f"report_{ts}.txt")
    json_path = os.path.join(out_dir, "backtest_latest.json")

    # ── Build equity curve ────────────────────────────────────────────────
    equity_curve = []
    if len(combined_vals) > 0:
        strat_base = combined_vals.iloc[0]
        spy_t      = spy.reindex(combined_vals.index).dropna()

        # Try AGG for balanced curve
        try:
            agg_raw = yf.download(["AGG"], period="2y", auto_adjust=True, progress=False)
            if isinstance(agg_raw.columns, pd.MultiIndex):
                agg = agg_raw["Close"]["AGG"].dropna()
            else:
                agg = agg_raw["Close"].squeeze().dropna()
            agg_t = agg.reindex(combined_vals.index).dropna()
        except Exception:
            agg_t = pd.Series(dtype=float)

        spy_base = spy_t.iloc[0] if len(spy_t) > 0 else None

        # Pre-compute balanced returns for the curve
        bal_series = None
        if spy_base is not None and len(agg_t) > 0:
            idx_both = spy_t.index.intersection(agg_t.index)
            if len(idx_both) > 1:
                sr = spy_t.loc[idx_both].pct_change().fillna(0)
                ar = agg_t.loc[idx_both].pct_change().fillna(0)
                br = (0.6 * sr + 0.4 * ar)
                bal_series = (1 + br).cumprod() * 10000

        for date in combined_vals.index:
            strat_val = round(combined_vals.loc[date] / strat_base * 10000, 2)
            spy_val   = None
            if spy_base is not None and date in spy_t.index:
                spy_val = round(spy_t.loc[date] / spy_base * 10000, 2)
            bal_val = None
            if bal_series is not None and date in bal_series.index:
                bal_val = round(float(bal_series.loc[date]), 2)
            equity_curve.append({
                "date":     str(date.date()),
                "strategy": strat_val,
                "spy":      spy_val,
                "balanced": bal_val,
            })

    # ── Diagnostic (hardcoded from validated run) ─────────────────────────
    diagnostic = {
        "unconditional_hit_rate": 0.513,
        "rsi_35_hit_rate":        0.595,
        "rsi_35_delta_pp":        8.2,
        "rsi_35_verdict":         "HAS EDGE",
        "macd_hit_rate":          0.521,
        "macd_delta_pp":          0.8,
        "macd_verdict":           "NO EDGE",
        "trend5_hit_rate":        0.514,
        "trend5_delta_pp":        0.1,
        "trend5_verdict":         "NO EDGE",
        "note": (
            "Only RSI<35 has statistically validated predictive power (+8.2pp hit rate). "
            "MACD and 5-day trend showed no edge in OOS testing. When 3 bull signals agree, "
            "hit rate = 50% (baseline). The scoring system firing on MACD+trend agreement "
            "has no edge."
        ),
    }

    # ── Build JSON payload ────────────────────────────────────────────────
    def _win_row(r):
        return {
            "window":               r.get("window"),
            "train_start":          r.get("train_start"),
            "test_start":           r.get("test_start"),
            "test_end":             r.get("test_end"),
            "total_return":         r.get("total_return"),
            "sharpe":               r.get("sharpe"),
            "sortino":              r.get("sortino"),
            "calmar":               r.get("calmar"),
            "max_drawdown":         r.get("max_drawdown"),
            "max_dd_duration_days": r.get("max_dd_duration_days"),
            "n_trades":             r.get("n_trades"),
            "win_rate":             r.get("win_rate"),
            "avg_win_loss_ratio":   r.get("avg_win_loss_ratio"),
            "profit_factor":        r.get("profit_factor"),
            "alpha":                r.get("alpha"),
            "beta":                 r.get("beta"),
        }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_windows":    len(window_results),
        "walk_forward": {
            "windows":   [_win_row(r) for r in window_results],
            "aggregate": {
                "total_return":         agg_m.get("total_return"),
                "ann_return":           agg_m.get("ann_return"),
                "sharpe":               agg_m.get("sharpe"),
                "sortino":              agg_m.get("sortino"),
                "calmar":               agg_m.get("calmar"),
                "max_drawdown":         agg_m.get("max_drawdown"),
                "max_dd_duration_days": agg_m.get("max_dd_duration_days"),
                "n_trades":             agg_m.get("n_trades"),
                "win_rate":             agg_m.get("win_rate"),
                "avg_win_loss_ratio":   agg_m.get("avg_win_loss_ratio"),
                "profit_factor":        agg_m.get("profit_factor"),
                "alpha":                agg_m.get("alpha"),
                "beta":                 agg_m.get("beta"),
            },
        },
        "benchmarks": {
            "spy_bh":         spy_bh,
            "balanced_60_40": balanced,
            "random_baseline": mc_baseline if mc_baseline else {},
        },
        "equity_curve": equity_curve,
        "diagnostic":   diagnostic,
    }

    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # ── Write text report ─────────────────────────────────────────────────
    sep  = "=" * 70
    dash = "-" * 70

    lines = [
        sep,
        "  FridayTrader v3 — Walk-Forward Backtest Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
        "  AGGREGATE OUT-OF-SAMPLE (all WF windows combined)",
        dash,
        f"  Total Return      : {(agg_m.get('total_return') or 0)*100:>+8.2f}%   "
        f"Ann: {(agg_m.get('ann_return') or 0)*100:>+8.2f}%",
        f"  Sharpe            : {agg_m.get('sharpe') or 0:>8.3f}   "
        f"Sortino: {agg_m.get('sortino') or 0:>8.3f}",
        f"  Calmar            : {agg_m.get('calmar') or 'N/A':>8}",
        f"  Max Drawdown      : {(agg_m.get('max_drawdown') or 0)*100:>8.2f}%   "
        f"(dur: {agg_m.get('max_dd_duration_days') or 0} days)",
        f"  # Trades          : {agg_m.get('n_trades') or 0:>8}",
        f"  Win Rate          : {(agg_m.get('win_rate') or 0)*100:>8.1f}%",
        f"  Profit Factor     : {agg_m.get('profit_factor') or 'N/A':>8}",
        f"  Alpha vs SPY      : {(agg_m.get('alpha') or 0)*100:>+8.2f}%   "
        f"Beta: {agg_m.get('beta') or 'N/A'}",
        "",
        "  WALK-FORWARD WINDOWS",
        dash,
        f"  {'Win':>3}  {'Test Period':<24}  {'Ret':>7}  {'Sharpe':>7}  "
        f"{'MaxDD':>7}  {'WinR':>6}  {'#Tr':>4}",
        f"  {'─'*3}  {'─'*24}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*4}",
    ]

    for r in window_results:
        marker = ("✓" if r.get("sharpe", 0) > 0.5
                  else ("✗" if r.get("sharpe", 0) < 0 else "~"))
        wr = (r.get("win_rate") or 0) * 100
        lines.append(
            f"  {marker}{r['window']:>2}  "
            f"{r.get('test_start','')!s} – {r.get('test_end','')!s}  "
            f"{r.get('total_return', 0)*100:>+6.1f}%  "
            f"{r.get('sharpe', 0):>7.2f}  "
            f"{r.get('max_drawdown', 0)*100:>6.1f}%  "
            f"{wr:>5.1f}%  {r.get('n_trades', 0):>4}"
        )

    lines += [
        "",
        "  BENCHMARKS (over combined WF OOS period)",
        dash,
        f"  SPY B&H       : return={spy_bh.get('total_return', 0)*100:>+.1f}%  "
        f"Sharpe={spy_bh.get('sharpe', 0):.2f}  "
        f"MaxDD={spy_bh.get('max_drawdown', 0)*100:.1f}%",
        f"  60/40 Balanced: return={balanced.get('total_return', 0)*100:>+.1f}%  "
        f"Sharpe={balanced.get('sharpe', 0):.2f}  "
        f"MaxDD={balanced.get('max_drawdown', 0)*100:.1f}%",
        "",
    ]

    if mc_baseline:
        lines += [
            "  MONTE CARLO RANDOM BASELINE (100 runs)",
            dash,
            f"  Sharpe  p10/p50/p90 : "
            f"{mc_baseline.get('sharpe_p10', 0):.2f} / "
            f"{mc_baseline.get('sharpe_p50', 0):.2f} / "
            f"{mc_baseline.get('sharpe_p90', 0):.2f}",
            f"  Return  p10/p50/p90 : "
            f"{mc_baseline.get('return_p10', 0)*100:.1f}% / "
            f"{mc_baseline.get('return_p50', 0)*100:.1f}% / "
            f"{mc_baseline.get('return_p90', 0)*100:.1f}%",
            "",
        ]
    else:
        lines += ["  MONTE CARLO RANDOM BASELINE: skipped (run without --fast)", ""]

    lines += [
        "  SIGNAL DIAGNOSTIC FINDINGS",
        dash,
        f"  Unconditional hit rate  : {diagnostic['unconditional_hit_rate']*100:.1f}%",
        f"  RSI<35 hit rate         : {diagnostic['rsi_35_hit_rate']*100:.1f}%  "
        f"(+{diagnostic['rsi_35_delta_pp']:.1f}pp)  → {diagnostic['rsi_35_verdict']}",
        f"  MACD hit rate           : {diagnostic['macd_hit_rate']*100:.1f}%  "
        f"(+{diagnostic['macd_delta_pp']:.1f}pp)  → {diagnostic['macd_verdict']}",
        f"  5d-trend hit rate       : {diagnostic['trend5_hit_rate']*100:.1f}%  "
        f"(+{diagnostic['trend5_delta_pp']:.1f}pp)  → {diagnostic['trend5_verdict']}",
        "",
        f"  NOTE: {diagnostic['note']}",
        "",
        sep,
    ]

    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nReports saved:")
    print(f"  Text : {txt_path}")
    print(f"  JSON : {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    fast_mode = "--fast" in sys.argv  # skip Monte Carlo if --fast

    closes, opens, highs, lows, volumes, spy = download_data()

    print("\nComputing ATR matrix...")
    atrs = compute_atr_matrix(closes, highs, lows)

    # ── Full in-sample / OOS split (kept for reference) ──────────────────
    n         = len(closes)
    split_idx = int(n * TRAIN_FRAC)
    train_idx = closes.index[:split_idx]
    test_idx  = closes.index[split_idx:]
    split_date = closes.index[split_idx]
    train_closes = closes.loc[train_idx]

    print(f"\nFull split: IS {closes.index[0].date()} – {train_idx[-1].date()} | "
          f"OOS {split_date.date()} – {closes.index[-1].date()}")

    # Train for full-split reference
    raw_pairs = find_cointegrated_pairs(train_closes)
    pairs     = fit_pairs_betas(train_closes, raw_pairs)
    ml_model, cv_auc = train_ml_model(train_closes, verbose=True)

    tech    = compute_tech_scores(closes, volumes)
    regime  = compute_regime_scores(closes)
    pair_s  = compute_pairs_scores(closes, pairs)
    ml_s    = compute_ml_scores(closes, ml_model)
    blended = blend(tech, regime, pair_s, ml_s, weights=ADVANCED_LAYER_WEIGHTS)

    is_vals,  is_trades  = simulate(train_idx, closes, opens, blended, atrs)
    oos_vals, oos_trades = simulate(test_idx,  closes, opens, blended, atrs)

    spy_is  = spy.reindex(train_idx).dropna()
    spy_oos = spy.reindex(test_idx).dropna()

    is_m  = full_metrics(is_vals,  is_trades,  spy_is,  "IN-SAMPLE")
    oos_m = full_metrics(oos_vals, oos_trades, spy_oos, "OUT-OF-SAMPLE")

    # Print full-split results
    w = 50
    print(f"\n{'='*w}\n  FULL-SPLIT BACKTEST RESULTS\n{'='*w}")
    for m in [is_m, oos_m]:
        lbl = m.get('label', '?')
        print(f"\n  ── {lbl} ──")
        print(f"  Return      : {m.get('total_return', 0)*100:+.1f}%   Ann: {m.get('ann_return', 0)*100:+.1f}%")
        print(f"  Sharpe      : {m.get('sharpe', 0):.2f}   Sortino: {m.get('sortino', 0):.2f}   Calmar: {m.get('calmar') or 'N/A'}")
        print(f"  Max Drawdown: {m.get('max_drawdown', 0)*100:.1f}%  (dur: {m.get('max_dd_duration_days', 0)} days)")
        print(f"  Trades      : {m.get('n_trades', 0)}  Win%: {(m.get('win_rate') or 0)*100:.1f}%  Profit factor: {m.get('profit_factor') or 'N/A'}")
        print(f"  Alpha vs SPY: {(m.get('alpha') or 0)*100:+.2f}%  Beta: {m.get('beta') or 'N/A'}")

    spy_bh_m = benchmark_spy_bh(spy, test_idx)
    print(f"\n  SPY B&H (OOS): return={spy_bh_m.get('total_return', 0)*100:+.1f}%  Sharpe={spy_bh_m.get('sharpe', 0):.2f}")

    # ── Walk-forward ──────────────────────────────────────────────────────
    wf_results, combined_vals, all_wf_trades = walk_forward(
        closes, opens, highs, lows, volumes, spy, atrs
    )

    spy_combined = spy.reindex(combined_vals.index).dropna()
    agg_m = full_metrics(combined_vals, all_wf_trades, spy_combined, "WF AGGREGATE OOS")

    print(f"\n{'='*w}\n  WALK-FORWARD AGGREGATE (OOS only)\n{'='*w}")
    print(f"  Return      : {agg_m.get('total_return', 0)*100:+.1f}%")
    print(f"  Sharpe      : {agg_m.get('sharpe', 0):.2f}   Sortino: {agg_m.get('sortino', 0):.2f}")
    print(f"  Max Drawdown: {agg_m.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Win Rate    : {(agg_m.get('win_rate') or 0)*100:.1f}%   Profit Factor: {agg_m.get('profit_factor') or 'N/A'}")
    print(f"  Alpha       : {(agg_m.get('alpha') or 0)*100:+.2f}%   Beta: {agg_m.get('beta') or 'N/A'}")

    # Walk-forward table
    print(f"\n  {'Win':>2}  {'Period':<22}  {'Ret':>7}  {'Sharpe':>7}  {'MaxDD':>7}  {'WinR':>6}  {'#Tr':>4}")
    print(f"  {'─'*2}  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*4}")
    for r in wf_results:
        marker = "✓" if r.get("sharpe", 0) > 0.5 else ("✗" if r.get("sharpe", 0) < 0 else "~")
        wr = (r.get("win_rate") or 0) * 100
        print(f"  {marker}  {r['test_start']} – {r['test_end']}  "
              f"{r.get('total_return', 0)*100:>+6.1f}%  "
              f"{r.get('sharpe', 0):>7.2f}  "
              f"{r.get('max_drawdown', 0)*100:>6.1f}%  "
              f"{wr:>5.1f}%  {r.get('n_trades', 0):>4}")

    # ── Benchmarks ───────────────────────────────────────────────────────
    print(f"\n{'─'*w}\n  BENCHMARKS (over combined WF OOS period)\n{'─'*w}")
    spy_bh_wf = benchmark_spy_bh(spy, combined_vals.index)
    bal_wf    = benchmark_60_40(spy, combined_vals.index)
    print(f"  SPY B&H    : return={spy_bh_wf.get('total_return', 0)*100:+.1f}%  Sharpe={spy_bh_wf.get('sharpe', 0):.2f}")
    print(f"  60/40      : return={bal_wf.get('total_return', 0)*100:+.1f}%  Sharpe={bal_wf.get('sharpe', 0):.2f}")

    # ── Monte Carlo ──────────────────────────────────────────────────────
    mc = {}
    if not fast_mode:
        mc = monte_carlo_baseline(closes, opens, atrs, combined_vals.index, n_runs=100)
        print(f"\n  Random baseline (100 runs, OOS):")
        print(f"  Sharpe p10/p50/p90: {mc.get('sharpe_p10', 0):.2f} / {mc.get('sharpe_p50', 0):.2f} / {mc.get('sharpe_p90', 0):.2f}")
        if agg_m.get("sharpe", 0) < mc.get("sharpe_p50", 0):
            print(f"  ⚠  Strategy Sharpe ({agg_m.get('sharpe', 0):.2f}) is BELOW the random median "
                  f"({mc.get('sharpe_p50', 0):.2f}) — strategy does not beat dart-throwing.")
        else:
            print(f"  ✓  Strategy Sharpe ({agg_m.get('sharpe', 0):.2f}) beats random median "
                  f"({mc.get('sharpe_p50', 0):.2f}).")
    else:
        print("\n  (Monte Carlo skipped — run without --fast to include)")

    # ── Layer ablation ────────────────────────────────────────────────────
    walk_forward_layer_analysis(closes, opens, atrs, tech, regime, pair_s, ml_s, test_idx, split_date)

    # ── Save report ───────────────────────────────────────────────────────
    save_report(wf_results, agg_m, spy_bh_wf, bal_wf, mc, closes, combined_vals, spy)

    try:
        import quantstats as qs
        full_rets = combined_vals.pct_change().dropna()
        spy_bench = spy.reindex(full_rets.index).pct_change().dropna().reindex(full_rets.index).fillna(0)
        qs.reports.html(full_rets, benchmark=spy_bench,
                        output="backtest_report.html",
                        title="FridayTrader v3 Walk-Forward Backtest")
        print("\nquantstats HTML report → backtest_report.html")
    except ImportError:
        print("\n(pip install quantstats for HTML tearsheet)")
    except Exception as e:
        print(f"\n(quantstats: {e})")
