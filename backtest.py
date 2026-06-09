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

STARTING_CASH  = 10_000
ROUND_TRIP_COST = 0.003   # 0.10% commission + 0.20% slippage per fill; ~0.60% round-trip
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
        vals = simulate(test_idx, closes, opens, blended, atrs)
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    closes, opens, highs, lows, volumes, spy = download_data()

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
    print("  Computing ATR matrix...")
    atrs = compute_atr_matrix(closes, highs, lows)
    print("  Done.")

    # ── Simulations ───────────────────────────────────────────────────────
    print("\nRunning in-sample simulation...")
    is_vals  = simulate(train_idx, closes, opens, blended, atrs)

    print("Running out-of-sample simulation...")
    oos_vals = simulate(test_idx, closes, opens, blended, atrs)

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

    # ── Walk-forward layer ablation ───────────────────────────────────────
    walk_forward_layer_analysis(closes, opens, atrs, tech, regime, pair_s, ml_s, test_idx, split_date)

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
