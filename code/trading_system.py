#!/usr/bin/env python3
"""Simple ETF momentum trading system with backtest and order generation.

Universe: QQQ, GLD, and SGOV.
Data source: Stooq daily CSV via HTTP (no API key).

Outputs:
- report/backtest_summary.json
- report/equity_curve.csv
- report/latest_orders.json
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
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

START_CAPITAL = 50_000.0
MAX_DRAWDOWN_TARGET = 0.20
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "report"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
}
RISK_ASSETS = ["QQQ", "GLD"]
SAFE_ASSET = "SGOV"  # 0-3 month Treasury ETF proxy for cash-like parking
UNIVERSE = RISK_ASSETS + [SAFE_ASSET]
BENCHMARK = "QQQ"
TRADING_DAYS_PER_YEAR = 252
MOMENTUM_WEIGHTS = {21: 0.25, 63: 0.35, 126: 0.40}
TREND_WINDOW = 200
VOL_WINDOW = 20
EPSILON = 0.0001
REBALANCE_EVERY_N_DAYS = 21
TOP_N = 1
STOP_LOSS_VOL_MULTIPLIER = 2.3
MIN_STOP_LOSS_PCT = 0.10
MAX_STOP_LOSS_PCT = 0.22
DISABLE_HARD_TAKE_PROFIT = True
HARD_TAKE_PROFIT_PCT = 0.60


@dataclass
class Bar:
    d: date
    close: float


PriceMap = Dict[str, List[float]]
Holdings = Dict[str, float]


def fetch_text(url: str) -> str:
    return urlopen(Request(url, headers=REQUEST_HEADERS), timeout=30).read().decode("utf-8")


def fetch_stooq(symbol: str) -> List[Bar]:
    symbol_key = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol_key}&i=d"
    text = fetch_text(url)
    rows = list(csv.DictReader(text.splitlines()))
    bars: List[Bar] = []
    for row in rows:
        try:
            bars.append(Bar(datetime.strptime(row["Date"], "%Y-%m-%d").date(), float(row["Close"])))
        except Exception:
            continue
    return bars


def fetch_yahoo(symbol: str) -> List[Bar]:
    params = urlencode({
        "period1": 0,
        "period2": int(datetime.now().timestamp()),
        "interval": "1d",
        "events": "history",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"
    payload = json.loads(fetch_text(url))
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    timestamps = result[0].get("timestamp") or []
    closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close") or []
    bars = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        bars.append(Bar(datetime.fromtimestamp(ts).date(), float(close)))
    return bars


def fetch_prices(symbol: str) -> List[Bar]:
    bars = fetch_stooq(symbol)
    if bars:
        return bars
    bars = fetch_yahoo(symbol)
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


def align_prices(price_map: Dict[str, List[Bar]]) -> Tuple[List[date], PriceMap]:
    ds = intersection_dates(price_map)
    out: PriceMap = {}
    for sym, bars in price_map.items():
        mp = {b.d: b.close for b in bars}
        out[sym] = [mp[d] for d in ds]
    return ds, out


def score_symbol(sym: str, prices: PriceMap, i: int) -> float:
    s = prices[sym]
    if s[i] <= sma(s, TREND_WINDOW, i):
        return 0.0
    raw_momentum = sum(weight * pct_change(s, lookback, i) for lookback, weight in MOMENTUM_WEIGHTS.items())
    volatility = stdev_daily_returns(s, VOL_WINDOW, i) or EPSILON
    return raw_momentum / volatility


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dynamic_stop_pct(prices: PriceMap, sym: str, i: int) -> float:
    vol = stdev_daily_returns(prices[sym], VOL_WINDOW, i)
    return clamp(vol * math.sqrt(VOL_WINDOW) * STOP_LOSS_VOL_MULTIPLIER, MIN_STOP_LOSS_PCT, MAX_STOP_LOSS_PCT)


def drawdown_multiplier(dd: float) -> float:
    multiplier = 1.0
    if dd > 0.10:
        multiplier *= 0.85
    if dd > 0.15:
        multiplier *= 0.60
    if dd > 0.19:
        multiplier *= 0.30
    return multiplier


def target_weights(prices: PriceMap, i: int, equity: float, peak: float) -> Dict[str, float]:
    weights = {sym: 0.0 for sym in prices.keys()}
    dd = 0.0 if peak <= 0 else 1.0 - equity / peak

    ranked = sorted(((score_symbol(sym, prices, i), sym) for sym in RISK_ASSETS), reverse=True)
    selected = [sym for score, sym in ranked[:TOP_N] if score > 0]
    if not selected:
        weights[SAFE_ASSET] = 1.0
        return weights

    inv_vols = [(1.0 / (stdev_daily_returns(prices[sym], VOL_WINDOW, i) or EPSILON), sym) for sym in selected]
    total = sum(v for v, _ in inv_vols)
    gross = drawdown_multiplier(dd)
    for inv_vol, sym in inv_vols:
        weights[sym] = gross * inv_vol / total
    weights[SAFE_ASSET] = 1.0 - sum(weights.values())
    return weights


def performance_metrics(equities: List[float], start_capital: float) -> Dict[str, float]:
    daily_returns = [equities[i] / equities[i - 1] - 1.0 for i in range(1, len(equities))]
    years = max(len(equities) / TRADING_DAYS_PER_YEAR, 1e-9)
    total_return = equities[-1] / start_capital - 1.0
    cagr = (equities[-1] / start_capital) ** (1 / years) - 1.0
    daily_vol = statistics.pstdev(daily_returns) if daily_returns else 0.0
    vol = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = 0.0
    if daily_vol > 0:
        sharpe = statistics.mean(daily_returns) / daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe": sharpe,
    }


def drawdown_curve(equities: List[float]) -> List[float]:
    peak = equities[0]
    drawdowns = []
    for equity in equities:
        peak = max(peak, equity)
        drawdowns.append(0.0 if peak <= 0 else 1.0 - equity / peak)
    return drawdowns


def compute_benchmark_metrics(dates: List[date], prices: PriceMap, start_i: int) -> Dict[str, object]:
    benchmark_prices = prices[BENCHMARK][start_i:]
    benchmark_equity = [START_CAPITAL * px / benchmark_prices[0] for px in benchmark_prices]
    benchmark_drawdowns = drawdown_curve(benchmark_equity)
    benchmark_curve = []
    for d, eq, dd in zip(dates[start_i:], benchmark_equity, benchmark_drawdowns):
        benchmark_curve.append({
            "date": d.isoformat(),
            "benchmark_equity": round(eq, 2),
            "benchmark_drawdown": round(dd, 6),
        })
    metrics = performance_metrics(benchmark_equity, START_CAPITAL)
    max_dd = max(benchmark_drawdowns)
    return {
        "curve": benchmark_curve,
        "summary": {
            "symbol": BENCHMARK,
            "end_equity": round(benchmark_equity[-1], 2),
            "total_return": round(metrics["total_return"], 4),
            "cagr": round(metrics["cagr"], 4),
            "annualized_volatility": round(metrics["annualized_volatility"], 4),
            "sharpe": round(metrics["sharpe"], 4),
            "max_drawdown": round(max_dd, 4),
        },
    }


def compute_dca_benchmark(dates: List[date], prices: PriceMap, start_i: int) -> Dict[str, object]:
    benchmark_prices = prices[BENCHMARK]
    months = sorted({(d.year, d.month) for d in dates[start_i:]})
    monthly_contribution = START_CAPITAL / len(months)
    remaining_cash = START_CAPITAL
    shares = 0.0
    contributed_months = set()
    dca_equity = []

    for i in range(start_i, len(dates)):
        month_key = (dates[i].year, dates[i].month)
        if month_key not in contributed_months:
            contribution = min(monthly_contribution, remaining_cash)
            shares += contribution / benchmark_prices[i]
            remaining_cash -= contribution
            contributed_months.add(month_key)
        dca_equity.append(remaining_cash + shares * benchmark_prices[i])

    dca_drawdowns = drawdown_curve(dca_equity)
    dca_curve = []
    for d, eq, dd in zip(dates[start_i:], dca_equity, dca_drawdowns):
        dca_curve.append({
            "date": d.isoformat(),
            "dca_equity": round(eq, 2),
            "dca_drawdown": round(dd, 6),
        })
    metrics = performance_metrics(dca_equity, START_CAPITAL)
    return {
        "curve": dca_curve,
        "summary": {
            "symbol": BENCHMARK,
            "method": "monthly_equal_dollar_dca",
            "total_principal": START_CAPITAL,
            "monthly_contribution": round(monthly_contribution, 2),
            "end_equity": round(dca_equity[-1], 2),
            "total_return": round(metrics["total_return"], 4),
            "cagr": round(metrics["cagr"], 4),
            "annualized_volatility": round(metrics["annualized_volatility"], 4),
            "sharpe": round(metrics["sharpe"], 4),
            "max_drawdown": round(max(dca_drawdowns), 4),
        },
    }


def portfolio_value(cash: float, holdings: Holdings, prices: PriceMap, i: int) -> float:
    return cash + sum(holdings[sym] * prices[sym][i] for sym in holdings)


def reset_position(
    sym: str,
    entry_prices: Dict[str, float],
    highest_prices: Dict[str, float],
    stop_lines: Dict[str, float],
    take_profit_lines: Dict[str, float],
) -> None:
    entry_prices[sym] = 0.0
    highest_prices[sym] = 0.0
    stop_lines[sym] = 0.0
    take_profit_lines[sym] = 0.0


def apply_daily_exits(
    prices: PriceMap,
    i: int,
    holdings: Holdings,
    cash: float,
    entry_prices: Dict[str, float],
    highest_prices: Dict[str, float],
    stop_lines: Dict[str, float],
    take_profit_lines: Dict[str, float],
) -> float:
    for sym, shares in holdings.items():
        if sym == SAFE_ASSET or shares <= 0:
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
            cash += shares * px
            holdings[sym] = 0.0
            reset_position(sym, entry_prices, highest_prices, stop_lines, take_profit_lines)
    return cash


def rebalance_portfolio(
    prices: PriceMap,
    i: int,
    equity: float,
    weights: Dict[str, float],
    previous_holdings: Holdings,
    entry_prices: Dict[str, float],
    highest_prices: Dict[str, float],
    stop_lines: Dict[str, float],
    take_profit_lines: Dict[str, float],
) -> Tuple[Holdings, float]:
    new_holdings: Holdings = {}
    spent = 0.0
    for sym, series in prices.items():
        px = series[i]
        shares = math.floor(equity * weights.get(sym, 0.0) / px) if px > 0 else 0
        new_holdings[sym] = float(shares)
        spent += shares * px

        if shares > 0:
            if previous_holdings.get(sym, 0.0) <= 0:
                entry_prices[sym] = px
                highest_prices[sym] = px
            else:
                highest_prices[sym] = max(highest_prices[sym], px)
            stop_pct = dynamic_stop_pct(prices, sym, i)
            stop_lines[sym] = highest_prices[sym] * (1.0 - stop_pct)
            take_profit_lines[sym] = (
                0.0 if DISABLE_HARD_TAKE_PROFIT else entry_prices[sym] * (1.0 + HARD_TAKE_PROFIT_PCT)
            )
        else:
            reset_position(sym, entry_prices, highest_prices, stop_lines, take_profit_lines)

    return new_holdings, equity - spent


def build_orders(prices: PriceMap, i: int, equity: float, weights: Dict[str, float]) -> List[Dict[str, object]]:
    orders = []
    for sym, weight in weights.items():
        if weight <= 0:
            continue
        px = prices[sym][i]
        qty = math.floor(equity * weight / px)
        if qty <= 0:
            continue
        stop_pct = dynamic_stop_pct(prices, sym, i)
        order = {
            "symbol": sym,
            "side": "BUY",
            "qty": qty,
            "est_price": round(px, 2),
            "est_value": round(qty * px, 2),
            "target_weight": round(weight, 4),
            "stop_loss_pct": round(stop_pct, 4),
            "stop_loss_price": round(px * (1.0 - stop_pct), 2),
        }
        if not DISABLE_HARD_TAKE_PROFIT:
            order["take_profit_pct"] = round(HARD_TAKE_PROFIT_PCT, 4)
            order["take_profit_price"] = round(px * (1.0 + HARD_TAKE_PROFIT_PCT), 2)
        orders.append(order)
    return orders


def backtest() -> Dict[str, object]:
    symbols = sorted(set(UNIVERSE + [SAFE_ASSET]))
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    raw = {sym: fetch_prices(sym) for sym in symbols}
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

        cash = apply_daily_exits(
            prices, i, holdings, cash, entry_prices, highest_prices, stop_lines, take_profit_lines
        )
        equity = portfolio_value(cash, holdings, prices, i)
        peak = max(peak, equity)
        if i - last_rebalance >= REBALANCE_EVERY_N_DAYS:
            weights = target_weights(prices, i, equity, peak)
            previous_holdings = holdings
            holdings, cash = rebalance_portfolio(
                prices, i, equity, weights, previous_holdings,
                entry_prices, highest_prices, stop_lines, take_profit_lines
            )
            last_rebalance = i
            last_weights = weights
            equity = portfolio_value(cash, holdings, prices, i)
            peak = max(peak, equity)

        equity = portfolio_value(cash, holdings, prices, i)
        dd = 0.0 if peak <= 0 else 1.0 - equity / peak
        row = {
            "date": d.isoformat(),
            "equity": round(equity, 2),
            "drawdown": round(dd, 6),
            **{f"w_{sym}": round(last_weights.get(sym, 0.0), 4) for sym in symbols},
        }
        for sym in symbols:
            if last_weights.get(sym, 0.0) > 0:
                row[f"stop_{sym}"] = round(stop_lines.get(sym, 0.0), 4)
                row[f"tp_{sym}"] = round(take_profit_lines.get(sym, 0.0), 4)
        equity_curve.append(row)

    benchmark = compute_benchmark_metrics(dates, prices, start_i)
    dca_benchmark = compute_dca_benchmark(dates, prices, start_i)
    for row, bench_row, dca_row in zip(equity_curve, benchmark["curve"], dca_benchmark["curve"]):
        row["benchmark_equity"] = bench_row["benchmark_equity"]
        row["benchmark_drawdown"] = bench_row["benchmark_drawdown"]
        row["dca_equity"] = dca_row["dca_equity"]
        row["dca_drawdown"] = dca_row["dca_drawdown"]
    save_csv(REPORT_DIR / "equity_curve.csv", equity_curve, list(equity_curve[0].keys()))

    equities = [row["equity"] for row in equity_curve]
    metrics = performance_metrics(equities, START_CAPITAL)
    max_dd = max(row["drawdown"] for row in equity_curve)

    latest_i = len(dates) - 1
    latest_equity = equities[-1]
    latest_weights = target_weights(prices, latest_i, latest_equity, max(equities))
    orders = build_orders(prices, latest_i, latest_equity, latest_weights)
    with (REPORT_DIR / "latest_orders.json").open("w") as f:
        json.dump({"as_of": dates[-1].isoformat(), "starting_capital": START_CAPITAL, "orders": orders}, f, indent=2)

    benchmark_summary = benchmark["summary"]
    dca_summary = dca_benchmark["summary"]
    take_profit_text = "Disabled by default to avoid cutting long trends too early"
    if not DISABLE_HARD_TAKE_PROFIT:
        take_profit_text = f"Fixed {HARD_TAKE_PROFIT_PCT:.0%} hard take-profit from entry price"

    summary = {
        "as_of": dates[-1].isoformat(),
        "start_capital": START_CAPITAL,
        "end_equity": round(equities[-1], 2),
        "total_return": round(metrics["total_return"], 4),
        "cagr": round(metrics["cagr"], 4),
        "annualized_volatility": round(metrics["annualized_volatility"], 4),
        "sharpe": round(metrics["sharpe"], 4),
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_target": MAX_DRAWDOWN_TARGET,
        "meets_drawdown_target": max_dd <= MAX_DRAWDOWN_TARGET,
        "beats_benchmark_total_return": metrics["total_return"] > benchmark_summary["total_return"],
        "beats_benchmark_cagr": metrics["cagr"] > benchmark_summary["cagr"],
        "max_drawdown_below_benchmark": max_dd < benchmark_summary["max_drawdown"],
        "benchmark": benchmark_summary,
        "dca_benchmark": dca_summary,
        "beats_dca_total_return": metrics["total_return"] > dca_summary["total_return"],
        "beats_dca_cagr": metrics["cagr"] > dca_summary["cagr"],
        "max_drawdown_below_dca": max_dd < dca_summary["max_drawdown"],
        "latest_target_weights": {k: round(v, 4) for k, v in latest_weights.items() if v > 0},
        "universe": UNIVERSE,
        "safe_asset": SAFE_ASSET,
        "logic": {
            "ranking": "21/63/126-day weighted momentum divided by 20-day volatility, only above 200-day SMA",
            "selection": f"Top {TOP_N} ETF from QQQ/GLD; fallback to SGOV when no ETF qualifies",
            "risk": "ETF-only universe, drawdown governor, SGOV fallback, volatility-adaptive trailing stop-loss",
            "rebalance": f"Every {REBALANCE_EVERY_N_DAYS} trading days",
            "exit_bands": {
                "stop_loss": (
                    f"{VOL_WINDOW}-day volatility x {STOP_LOSS_VOL_MULTIPLIER}, "
                    f"clamped to {MIN_STOP_LOSS_PCT:.0%}-{MAX_STOP_LOSS_PCT:.0%}, "
                    "trailing from post-entry high"
                ),
                "take_profit": take_profit_text,
            },
            "assets": {
                "growth": "QQQ",
                "gold": "GLD",
                "defensive": SAFE_ASSET,
            },
        },
    }
    with (REPORT_DIR / "backtest_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    summary = backtest()
    print(json.dumps(summary, indent=2))
