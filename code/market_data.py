"""Market data loading and indicator helpers."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "prices"
REPORT_DIR = BASE_DIR / "report"
FACTOR_DIR = DATA_DIR / "factors"
FACTOR_PATH = FACTOR_DIR / "monthly_factors.csv"

MONTHLY_CONTRIBUTION = 10_000.0
MAX_DRAWDOWN_TARGET = 0.203
GROWTH_ASSET = "QQQ"
GOLD_ASSET = "GLD"
SAFE_ASSET = "SGOV"
UNIVERSE = [GROWTH_ASSET, GOLD_ASSET, SAFE_ASSET]

TRADING_DAYS_PER_YEAR = 252
FORWARD_RETURN_DAYS = 30
SLOW_TREND_WINDOW = 200
RSI_WINDOW = 14
MEDIUM_RETURN_WINDOW = 63

CACHE_MAX_AGE_SECONDS = 18 * 60 * 60
YAHOO_RETRY_DELAYS = [0, 2, 5, 10]
YAHOO_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
YAHOO_START_YEAR = 1990
YAHOO_CHUNK_YEARS = 10
FETCH_PAUSE_SECONDS = 1.0
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class Bar:
    d: date
    close: float


PriceMap = Dict[str, List[float]]
Holdings = Dict[str, float]


def fetch_text(url: str) -> str:
    """发起 HTTP GET 请求并返回文本；统一加请求头，降低 Yahoo/Stooq 拒绝请求的概率。"""

    return urlopen(Request(url, headers=REQUEST_HEADERS), timeout=30).read().decode("utf-8")


def cache_path(symbol: str) -> Path:
    """返回某个标的本地价格缓存文件路径。"""

    return CACHE_DIR / f"{symbol.upper()}.csv"


def load_price_cache(symbol: str) -> List[Bar]:
    """读取本地价格缓存；坏行直接跳过，避免单行数据污染整个回测。"""

    path = cache_path(symbol)
    if not path.exists():
        return []
    bars: List[Bar] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                bars.append(Bar(datetime.strptime(row["Date"], "%Y-%m-%d").date(), float(row["Close"])))
            except Exception:
                continue
    return bars


def save_price_cache(symbol: str, bars: List[Bar]) -> None:
    """把下载到的日线收盘价写入本地缓存，供后续回测和限流兜底使用。"""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path(symbol).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Close"])
        writer.writeheader()
        for bar in bars:
            writer.writerow({"Date": bar.d.isoformat(), "Close": f"{bar.close:.8f}"})


def cache_is_fresh(symbol: str) -> bool:
    """判断缓存是否还在有效期内；有效则优先使用本地数据，减少网络请求。"""

    path = cache_path(symbol)
    return path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE_SECONDS


def fetch_stooq(symbol: str) -> List[Bar]:
    """从 Stooq 下载日线价格，作为 Yahoo 数据失败或限流时的备用来源。"""

    text = fetch_text(f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d")
    bars: List[Bar] = []
    for row in csv.DictReader(text.splitlines()):
        try:
            bars.append(Bar(datetime.strptime(row["Date"], "%Y-%m-%d").date(), float(row["Close"])))
        except Exception:
            continue
    return bars


def yahoo_url(symbol: str, period1: int, period2: int) -> str:
    """拼接 Yahoo chart API 路径；用时间窗口分段下载，避免单次请求过大。"""

    params = urlencode({"period1": period1, "period2": period2, "interval": "1d", "events": "history"})
    return f"/v8/finance/chart/{symbol}?{params}"


def parse_yahoo_chart(payload: Dict[str, object]) -> List[Bar]:
    """解析 Yahoo chart JSON，只保留有收盘价的日期。"""

    result = payload.get("chart", {}).get("result") or []
    if not result:
        return []
    timestamps = result[0].get("timestamp") or []
    closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close") or []
    return [
        Bar(datetime.fromtimestamp(ts).date(), float(close))
        for ts, close in zip(timestamps, closes)
        if close is not None
    ]


def fetch_yahoo_window(symbol: str, period1: int, period2: int) -> List[Bar]:
    """下载 Yahoo 指定时间窗口数据；遇到 429 会按退避列表重试 query1/query2。"""

    path = yahoo_url(symbol, period1, period2)
    last_error = None
    for host in YAHOO_HOSTS:
        for delay in YAHOO_RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                return parse_yahoo_chart(json.loads(fetch_text(f"https://{host}{path}")))
            except HTTPError as exc:
                last_error = exc
                if exc.code == 400:
                    return []
                if exc.code != 429:
                    raise
    raise RuntimeError(f"Yahoo rate limit did not clear for {symbol}") from last_error


def fetch_yahoo(symbol: str) -> List[Bar]:
    """按多年窗口循环下载 Yahoo 历史价格，并按日期去重排序。"""

    bars = []
    current_year = datetime.now().year
    for start_year in range(YAHOO_START_YEAR, current_year + 1, YAHOO_CHUNK_YEARS):
        end_year = min(start_year + YAHOO_CHUNK_YEARS, current_year + 1)
        period1 = int(datetime(start_year, 1, 1).timestamp())
        period2 = int(datetime(end_year, 1, 1).timestamp())
        bars.extend(fetch_yahoo_window(symbol, period1, period2))
        time.sleep(FETCH_PAUSE_SECONDS)
    unique = {bar.d: bar for bar in bars}
    if not unique:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    return [unique[d] for d in sorted(unique)]


def fetch_prices(symbol: str) -> List[Bar]:
    """获取某个标的价格：优先新鲜缓存，其次 Yahoo/Stooq，最后退回旧缓存。"""

    cached_bars = load_price_cache(symbol)
    if cached_bars and cache_is_fresh(symbol):
        return cached_bars

    last_error = None
    for fetcher in (fetch_yahoo, fetch_stooq):
        try:
            bars = fetcher(symbol)
        except Exception as exc:
            last_error = exc
            continue
        if bars:
            save_price_cache(symbol, bars)
            return bars

    if cached_bars:
        return cached_bars
    raise RuntimeError(f"No data for {symbol}") from last_error


def load_market_data() -> Tuple[List[date], PriceMap]:
    """加载所有标的价格并对齐交易日，确保组合回测每天都有完整价格。"""

    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    raw = {}
    for index, sym in enumerate(UNIVERSE):
        if index > 0:
            time.sleep(FETCH_PAUSE_SECONDS)
        raw[sym] = fetch_prices(sym)
    return align_prices(raw)


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """保存字典列表为 CSV；自动追加 rows 中出现的新字段。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(fieldnames)
    seen = set(all_fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sma(series: List[float], window: int, i: int) -> float:
    """计算截至第 i 天的简单移动平均；历史不足时用已有价格均值。"""

    if i - window + 1 < 0:
        return series[i]
    sl = series[i - window + 1 : i + 1]
    return sum(sl) / len(sl)


def pct_change(series: List[float], lookback: int, i: int) -> float:
    """计算第 i 天相对 lookback 天前的涨跌幅，作为动量类特征。"""

    if i - lookback < 0 or series[i - lookback] == 0:
        return 0.0
    return series[i] / series[i - lookback] - 1.0


def rsi(series: List[float], window: int, i: int) -> float:
    """计算简化版 RSI；用于判断资产是否短期过热或超卖。"""

    if i - window < 1:
        return 50.0
    gains = []
    losses = []
    for j in range(i - window + 1, i + 1):
        change = series[j] - series[j - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def intersection_dates(price_map: Dict[str, List[Bar]]) -> List[date]:
    """取所有标的共同拥有价格的日期，防止某个资产缺价导致组合估值错位。"""

    common = None
    for bars in price_map.values():
        ds = {b.d for b in bars}
        common = ds if common is None else common & ds
    return sorted(common)


def align_prices(price_map: Dict[str, List[Bar]]) -> Tuple[List[date], PriceMap]:
    """把每个标的价格映射到共同日期序列，输出统一 dates 和价格矩阵。"""

    ds = intersection_dates(price_map)
    out: PriceMap = {}
    for sym, bars in price_map.items():
        mp = {b.d: b.close for b in bars}
        out[sym] = [mp[d] for d in ds]
    return ds, out


def monthly_price_indices(dates: List[date], start_i: int) -> List[int]:
    """返回从 start_i 开始每个月第一个可交易日索引，作为月度预测和定投日期。"""

    seen = set()
    indices = []
    for i in range(start_i, len(dates)):
        key = (dates[i].year, dates[i].month)
        if key not in seen:
            seen.add(key)
            indices.append(i)
    return indices


def portfolio_value(holdings: Holdings, prices: PriceMap, i: int) -> float:
    """按第 i 天价格计算当前持仓总市值。"""

    return sum(holdings[sym] * prices[sym][i] for sym in holdings)


def rebalance(holdings: Holdings, prices: PriceMap, i: int, equity: float, weights: Dict[str, float]) -> Holdings:
    """按目标权重把组合再平衡为新的股数；这里假设无交易成本和可买零碎股。"""

    return {
        sym: (equity * weights.get(sym, 0.0) / prices[sym][i] if prices[sym][i] > 0 else 0.0)
        for sym in holdings
    }
