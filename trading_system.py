#!/usr/bin/env python3
"""Simple multi-asset momentum trading system with backtest and order generation.

Universe: liquid US growth / broad-market names.
Data source: Stooq daily CSV via HTTP (no API key).

Outputs:
- reports/backtest_summary.json
- reports/equity_curve.csv
- reports/latest_orders.json
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

START_CAPITAL = 50_000.0
MAX_DRAWDOWN_TARGET = 0.20
DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
UNIVERSE = [
    "SPY", "QQQ", "GOOGL", "META", "MSFT", "AMZN", "NVDA", "AVGO", "AAPL", "PLTR",
]
SAFE_ASSET = "SHY"  # short treasury ETF proxy for cash-like parking
BENCHMARK = "QQQ"
REBALANCE_EVERY_N_DAYS = 21
TOP_N = 4
STOP_LOSS_VOL_MULTIPLIER = 2.3
MIN_STOP_LOSS_PCT = 0.10
MAX_STOP_LOSS_PCT = 0.22
PARTIAL_REGIME_EXPOSURE = 0.45
DISABLE_HARD_TAKE_PROFIT = True
HARD_TAKE_PROFIT_PCT = 0.60


@dataclass
class Bar:
    d: date
    close: float


def fetch_stooq(symbol: str) -> List[Bar]:
    symbol_key = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol_key}&i=d"
    text = urlopen(url, timeout=30).read().decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    bars: List[Bar] = []
    for row in rows:
        try:
            bars.append(Bar(datetime.strptime(row["Date"], "%Y-%m-%d").date(), float(row["Close"])))
        except Exception:
            continue
    if not bars:
        raise RuntimeError(f"No data for {symbol}")
    return bars


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(fieldnames)
    seen = set(all_fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def pct_change(series: List[float], lookback: int, i: int) -> float:
    if i - lookback < 0 or series[i - lookback] == 0:
        return 0.0
    return series[i] / series[i - lookback] - 1.0


def sma(series: List[float], window: int, i: int) -> float:
    if i - window + 1 < 0:
        return series[i]
    sl = series[i - window + 1 : i + 1]
    return sum(sl) / len(sl)


def stdev_daily_returns(series: List[float], window: int, i: int) -> float:
    if i - window < 1:
        return 0.0
    rets = [series[j] / series[j - 1] - 1.0 for j in range(i - window + 1, i + 1)]
    return statistics.pstdev(rets) if len(rets) > 1 else 0.0


def intersection_dates(price_map: Dict[str, List[Bar]]) -> List[date]:
    common = None
    for bars in price_map.values():
        ds = {b.d for b in bars}
        common = ds if common is None else common & ds
    return sorted(common)


def align_prices(price_map: Dict[str, List[Bar]]) -> Tuple[List[date], Dict[str, List[float]]]:
    ds = intersection_dates(price_map)
    out: Dict[str, List[float]] = {}
    for sym, bars in price_map.items():
        mp = {b.d: b.close for b in bars}
        out[sym] = [mp[d] for d in ds]
    return ds, out


def score_symbol(sym: str, prices: Dict[str, List[float]], i: int) -> float:
    s = prices[sym]
    m21 = pct_change(s, 21, i)
    m63 = pct_change(s, 63, i)
    m126 = pct_change(s, 126, i)
    trend_ok = 1.0 if s[i] > sma(s, 200, i) else 0.0
    vol = stdev_daily_returns(s, 20, i) or 0.0001
    # weighted momentum adjusted by short-term realized vol, only if trend positive
    raw = 0.25 * m21 + 0.35 * m63 + 0.40 * m126
    return (raw / vol) * trend_ok


def regime_exposure(prices: Dict[str, List[float]], i: int) -> float:
    q = prices[BENCHMARK]
    spy = prices["SPY"]
    q_above = q[i] > sma(q, 200, i)
    spy_above = spy[i] > sma(spy, 200, i)
    if q_above and spy_above:
        return 1.0
    if q_above or spy_above:
        return PARTIAL_REGIME_EXPOSURE
    return 0.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dynamic_stop_pct(prices: Dict[str, List[float]], sym: str, i: int) -> float:
    vol = stdev_daily_returns(prices[sym], 20, i)
    return clamp(vol * math.sqrt(20) * STOP_LOSS_VOL_MULTIPLIER, MIN_STOP_LOSS_PCT, MAX_STOP_LOSS_PCT)


def target_weights(prices: Dict[str, List[float]], i: int, equity: float, peak: float) -> Dict[str, float]:
    weights = {sym: 0.0 for sym in prices.keys()}
    dd = 0.0 if peak <= 0 else 1.0 - equity / peak
    regime_gross = regime_exposure(prices, i)
    if regime_gross <= 0:
        weights[SAFE_ASSET] = 1.0
        return weights

    ranked = []
    for sym in UNIVERSE:
        if sym in {"SPY", "QQQ"}:
            continue
        ranked.append((score_symbol(sym, prices, i), sym))
    ranked.sort(reverse=True)
    selected = [sym for score, sym in ranked[:TOP_N] if score > 0]
    if not selected:
        weights[SAFE_ASSET] = 1.0
        return weights

    inv_vols = []
    for sym in selected:
        vol = stdev_daily_returns(prices[sym], 20, i) or 0.0001
        inv_vols.append((1.0 / vol, sym))
    total = sum(v for v, _ in inv_vols)
    gross = regime_gross
    if dd > 0.10:
        gross *= 0.85
    if dd > 0.15:
        gross *= 0.60
    if dd > 0.19:
        gross *= 0.30
    for inv_vol, sym in inv_vols:
        weights[sym] = gross * inv_vol / total
    weights[SAFE_ASSET] = 1.0 - sum(weights.values())
    return weights


def compute_benchmark_metrics(dates: List[date], prices: Dict[str, List[float]], start_i: int) -> Dict[str, object]:
    benchmark_prices = prices[BENCHMARK][start_i:]
    benchmark_equity = [START_CAPITAL * px / benchmark_prices[0] for px in benchmark_prices]
    peak = benchmark_equity[0]
    benchmark_curve = []
    for d, eq in zip(dates[start_i:], benchmark_equity):
        peak = max(peak, eq)
        benchmark_curve.append({
            "date": d.isoformat(),
            "benchmark_equity": round(eq, 2),
            "benchmark_drawdown": round(1.0 - eq / peak, 6),
        })
    daily_returns = [benchmark_equity[i] / benchmark_equity[i - 1] - 1.0 for i in range(1, len(benchmark_equity))]
    years = max(len(benchmark_equity) / 252.0, 1e-9)
    total_return = benchmark_equity[-1] / START_CAPITAL - 1.0
    cagr = (benchmark_equity[-1] / START_CAPITAL) ** (1 / years) - 1.0
    vol = statistics.pstdev(daily_returns) * math.sqrt(252) if daily_returns else 0.0
    sharpe = (statistics.mean(daily_returns) / statistics.pstdev(daily_returns) * math.sqrt(252)) if len(daily_returns) > 1 and statistics.pstdev(daily_returns) > 0 else 0.0
    max_dd = max(row["benchmark_drawdown"] for row in benchmark_curve)
    return {
        "curve": benchmark_curve,
        "summary": {
            "symbol": BENCHMARK,
            "end_equity": round(benchmark_equity[-1], 2),
            "total_return": round(total_return, 4),
            "cagr": round(cagr, 4),
            "annualized_volatility": round(vol, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
        },
    }


def backtest() -> Dict[str, object]:
    symbols = sorted(set(UNIVERSE + [SAFE_ASSET]))
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    raw = {sym: fetch_stooq(sym) for sym in symbols}
    dates, prices = align_prices(raw)

    start_i = 252
    holdings = {sym: 0.0 for sym in symbols}
    cash = START_CAPITAL
    peak = START_CAPITAL
    equity_curve = []
    last_rebalance = -999
    last_weights = {sym: 0.0 for sym in symbols}
    entry_prices = {sym: 0.0 for sym in symbols}
    highest_prices = {sym: 0.0 for sym in symbols}
    stop_lines = {sym: 0.0 for sym in symbols}
    take_profit_lines = {sym: 0.0 for sym in symbols}

    for i in range(start_i, len(dates)):
        d = dates[i]

        # daily risk management on existing positions
        for sym in symbols:
            if sym == SAFE_ASSET or holdings[sym] <= 0:
                continue
            px = prices[sym][i]
            highest_prices[sym] = max(highest_prices[sym], px)
            stop_pct = dynamic_stop_pct(prices, sym, i)
            trailing_stop = highest_prices[sym] * (1.0 - stop_pct)
            take_profit_line = entry_prices[sym] * (1.0 + HARD_TAKE_PROFIT_PCT)
            stop_lines[sym] = trailing_stop
            take_profit_lines[sym] = 0.0 if DISABLE_HARD_TAKE_PROFIT else take_profit_line
            hard_take_profit_hit = (not DISABLE_HARD_TAKE_PROFIT) and px >= take_profit_line
            if px <= trailing_stop or hard_take_profit_hit:
                cash += holdings[sym] * px
                holdings[sym] = 0.0
                entry_prices[sym] = 0.0
                highest_prices[sym] = 0.0
                stop_lines[sym] = 0.0
                take_profit_lines[sym] = 0.0

        equity = cash + sum(holdings[sym] * prices[sym][i] for sym in symbols)
        peak = max(peak, equity)
        if i - last_rebalance >= REBALANCE_EVERY_N_DAYS:
            weights = target_weights(prices, i, equity, peak)
            # rebalance at close using whole shares
            target_values = {sym: equity * w for sym, w in weights.items()}
            new_holdings = {}
            spent = 0.0
            for sym in symbols:
                px = prices[sym][i]
                shares = math.floor(target_values[sym] / px) if px > 0 else 0
                new_holdings[sym] = float(shares)
                spent += shares * px
            previous_holdings = holdings
            holdings = new_holdings
            cash = equity - spent
            last_rebalance = i
            last_weights = weights
            for sym in symbols:
                px = prices[sym][i]
                if holdings[sym] > 0:
                    if previous_holdings.get(sym, 0.0) <= 0:
                        entry_prices[sym] = px
                        highest_prices[sym] = px
                    else:
                        highest_prices[sym] = max(highest_prices[sym], px)
                    stop_pct = dynamic_stop_pct(prices, sym, i)
                    stop_lines[sym] = highest_prices[sym] * (1.0 - stop_pct)
                    take_profit_lines[sym] = 0.0 if DISABLE_HARD_TAKE_PROFIT else entry_prices[sym] * (1.0 + HARD_TAKE_PROFIT_PCT)
                else:
                    entry_prices[sym] = 0.0
                    highest_prices[sym] = 0.0
                    stop_lines[sym] = 0.0
                    take_profit_lines[sym] = 0.0
            equity = cash + sum(holdings[sym] * prices[sym][i] for sym in symbols)
            peak = max(peak, equity)

        equity = cash + sum(holdings[sym] * prices[sym][i] for sym in symbols)
        dd = 0.0 if peak <= 0 else 1.0 - equity / peak
        equity_curve.append({
            "date": d.isoformat(),
            "equity": round(equity, 2),
            "drawdown": round(dd, 6),
            **{f"w_{sym}": round(last_weights.get(sym, 0.0), 4) for sym in symbols},
            **{f"stop_{sym}": round(stop_lines.get(sym, 0.0), 4) for sym in symbols if last_weights.get(sym, 0.0) > 0},
            **{f"tp_{sym}": round(take_profit_lines.get(sym, 0.0), 4) for sym in symbols if last_weights.get(sym, 0.0) > 0},
        })

    benchmark = compute_benchmark_metrics(dates, prices, start_i)
    for row, bench_row in zip(equity_curve, benchmark["curve"]):
        row["benchmark_equity"] = bench_row["benchmark_equity"]
        row["benchmark_drawdown"] = bench_row["benchmark_drawdown"]
    save_csv(REPORT_DIR / "equity_curve.csv", equity_curve, list(equity_curve[0].keys()))

    equities = [row["equity"] for row in equity_curve]
    daily_returns = [equities[i] / equities[i - 1] - 1.0 for i in range(1, len(equities))]
    total_return = equities[-1] / START_CAPITAL - 1.0
    years = max((len(equity_curve) / 252.0), 1e-9)
    cagr = equities[-1] / START_CAPITAL
    cagr = cagr ** (1 / years) - 1.0
    max_dd = max(row["drawdown"] for row in equity_curve)
    vol = statistics.pstdev(daily_returns) * math.sqrt(252) if daily_returns else 0.0
    sharpe = (statistics.mean(daily_returns) / statistics.pstdev(daily_returns) * math.sqrt(252)) if len(daily_returns) > 1 and statistics.pstdev(daily_returns) > 0 else 0.0

    latest_i = len(dates) - 1
    latest_equity = equities[-1]
    latest_weights = target_weights(prices, latest_i, latest_equity, max(equities))
    orders = []
    # create recommended orders from current holdings=all cash assumption
    for sym, w in latest_weights.items():
        if w <= 0:
            continue
        px = prices[sym][latest_i]
        qty = math.floor(latest_equity * w / px)
        if qty > 0:
            stop_pct = dynamic_stop_pct(prices, sym, latest_i)
            order = {
                "symbol": sym,
                "side": "BUY",
                "qty": qty,
                "est_price": round(px, 2),
                "est_value": round(qty * px, 2),
                "target_weight": round(w, 4),
                "stop_loss_pct": round(stop_pct, 4),
                "stop_loss_price": round(px * (1.0 - stop_pct), 2),
            }
            if not DISABLE_HARD_TAKE_PROFIT:
                order["take_profit_pct"] = round(HARD_TAKE_PROFIT_PCT, 4)
                order["take_profit_price"] = round(px * (1.0 + HARD_TAKE_PROFIT_PCT), 2)
            orders.append(order)
    with (REPORT_DIR / "latest_orders.json").open("w") as f:
        json.dump({"as_of": dates[-1].isoformat(), "starting_capital": START_CAPITAL, "orders": orders}, f, indent=2)

    benchmark_summary = benchmark["summary"]
    summary = {
        "as_of": dates[-1].isoformat(),
        "start_capital": START_CAPITAL,
        "end_equity": round(equities[-1], 2),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "annualized_volatility": round(vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_target": MAX_DRAWDOWN_TARGET,
        "meets_drawdown_target": max_dd <= MAX_DRAWDOWN_TARGET,
        "beats_benchmark_total_return": total_return > benchmark_summary["total_return"],
        "beats_benchmark_cagr": cagr > benchmark_summary["cagr"],
        "max_drawdown_below_benchmark": max_dd < benchmark_summary["max_drawdown"],
        "benchmark": benchmark_summary,
        "latest_target_weights": {k: round(v, 4) for k, v in latest_weights.items() if v > 0},
        "universe": UNIVERSE,
        "safe_asset": SAFE_ASSET,
        "logic": {
            "ranking": "21/63/126-day weighted momentum divided by 20-day volatility, only above 200-day SMA",
            "selection": f"Top {TOP_N} stocks from growth universe",
            "risk": "Partial regime filter (QQQ/SPY 200-day SMA), inverse volatility sizing, softer drawdown governor, SHY fallback, volatility-adaptive trailing stop-loss",
            "rebalance": f"Every {REBALANCE_EVERY_N_DAYS} trading days",
            "exit_bands": {
                "stop_loss": f"20-day volatility × {STOP_LOSS_VOL_MULTIPLIER}, clamped to {MIN_STOP_LOSS_PCT:.0%}-{MAX_STOP_LOSS_PCT:.0%}, trailing from post-entry high",
                "take_profit": "Disabled by default to avoid cutting long trends too early" if DISABLE_HARD_TAKE_PROFIT else f"Fixed {HARD_TAKE_PROFIT_PCT:.0%} hard take-profit from entry price",
            },
            "regime_exposure": {
                "risk_on": "100% risk budget when both QQQ and SPY are above 200-day SMA",
                "mixed": f"{PARTIAL_REGIME_EXPOSURE:.0%} risk budget when only one of QQQ/SPY is above 200-day SMA",
                "risk_off": "100% SHY when both QQQ and SPY are below 200-day SMA",
            },
        },
    }
    with (REPORT_DIR / "backtest_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    summary = backtest()
    print(json.dumps(summary, indent=2))
