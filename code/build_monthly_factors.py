"""Build monthly factor data from price history and public macro data."""
from __future__ import annotations

import csv
import re
from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from market_data import (
    Bar,
    CACHE_DIR,
    DATA_DIR,
    FACTOR_PATH,
    FORWARD_RETURN_DAYS,
    RSI_WINDOW,
    SLOW_TREND_WINDOW,
    UNIVERSE,
    align_prices,
    monthly_price_indices,
    pct_change,
    rsi,
    save_csv,
    sma,
)
from prediction_module import FACTOR_COLUMNS, forward_label_end_index

MACRO_DIR = DATA_DIR / "macro"
TBILL_PATH = MACRO_DIR / "DTB3.csv"
TBILL_TEXT_PATH = MACRO_DIR / "DTB3.txt"


def parse_optional_rate(text: str) -> Optional[float]:
    """解析 FRED 利率字段；空值或点号返回 None，百分比数值转成小数。"""

    value = text.strip()
    if not value or value == ".":
        return None
    return float(value) / 100.0


def parse_tbill_csv(path: Path) -> Dict[date, float]:
    """解析 FRED CSV 格式的 DTB3 利率文件。"""

    rates: Dict[date, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        date_col = "observation_date" if "observation_date" in (reader.fieldnames or []) else "DATE"
        value_col = "DTB3"
        for row in reader:
            raw = row.get(value_col, "")
            rate = parse_optional_rate(raw)
            if rate is None:
                continue
            d = datetime.strptime(row[date_col], "%Y-%m-%d").date()
            rates[d] = rate
    return rates


def parse_tbill_text(path: Path) -> Dict[date, float]:
    """解析 FRED 表格网页文本中的 DTB3 日期和利率。"""

    rates: Dict[date, float] = {}
    pattern = re.compile(r"#(\d{4}-\d{2}-\d{2})\|\s*([-.0-9]+|\.)")
    for raw_date, raw_rate in pattern.findall(path.read_text()):
        rate = parse_optional_rate(raw_rate)
        if rate is None:
            continue
        d = datetime.strptime(raw_date, "%Y-%m-%d").date()
        rates[d] = rate
    return rates


def load_tbill_rates(csv_path: Path = TBILL_PATH, text_path: Path = TBILL_TEXT_PATH) -> Dict[date, float]:
    """读取 FRED DTB3 日度 3 个月 T-bill 利率，优先 CSV，兜底网页文本。"""

    if csv_path.exists():
        return parse_tbill_csv(csv_path)
    if text_path.exists():
        return parse_tbill_text(text_path)
    return {}


def latest_rate_on_or_before(rates: Dict[date, float], d: date) -> Optional[float]:
    """取决策日当天或之前最近一个可用 T-bill 利率。"""

    if not rates:
        return None
    rate_dates = sorted(rates)
    pos = bisect_right(rate_dates, d) - 1
    return None if pos < 0 else rates[rate_dates[pos]]


def load_cached_bars() -> Dict[str, list]:
    """从本地价格缓存读取 QQQ/GLD/SGOV 日线；不触发网络请求。"""

    raw = {}
    for symbol in UNIVERSE:
        path = CACHE_DIR / f"{symbol}.csv"
        rows = []
        with path.open() as f:
            for row in csv.DictReader(f):
                rows.append(Bar(datetime.strptime(row["Date"], "%Y-%m-%d").date(), float(row["Close"])))
        raw[symbol] = rows
    return raw


def price_feature_row(prices: Dict[str, List[float]], asset: str, i: int) -> Dict[str, float]:
    """计算写入月度数据表的资产价格、200 日偏离、RSI 和 63 日涨幅。"""

    prefix = asset.lower()
    series = prices[asset]
    return {
        f"{prefix}_close": series[i],
        f"{prefix}_extension_200dma": series[i] / sma(series, SLOW_TREND_WINDOW, i) - 1.0,
        f"{prefix}_rsi_14": rsi(series, RSI_WINDOW, i),
        f"{prefix}_return_63d": pct_change(series, 63, i),
    }


def label_row(dates, prices: Dict[str, List[float]], asset: str, i: int) -> Dict[str, object]:
    """构造指定资产未来 30 个自然日端点收益、窗口均价收益和方向标签。"""

    prefix = asset.lower()
    label_end_i = forward_label_end_index(dates, i)
    if label_end_i is None:
        return {
            f"label_{prefix}_forward_30d_return": "",
            f"label_{prefix}_forward_30d_avg_return": "",
            f"label_{prefix}_forward_30d_up": "",
            f"label_{prefix}_forward_30d_avg_up": "",
        }
    forward_return = prices[asset][label_end_i] / prices[asset][i] - 1.0
    forward_window = prices[asset][i + 1 : label_end_i + 1]
    avg_return = sum(forward_window) / len(forward_window) / prices[asset][i] - 1.0
    return {
        f"label_{prefix}_forward_30d_return": round(forward_return, 8),
        f"label_{prefix}_forward_30d_avg_return": round(avg_return, 8),
        f"label_{prefix}_forward_30d_up": int(forward_return > 0),
        f"label_{prefix}_forward_30d_avg_up": int(avg_return > 0),
    }


def build_rows() -> List[Dict[str, object]]:
    """按每月第一个交易日生成完整 monthly_factors.csv 行。"""

    dates, prices = align_prices(load_cached_bars())
    tbill_rates = load_tbill_rates()
    start_i = SLOW_TREND_WINDOW + 52
    rows = []
    for i in monthly_price_indices(dates, start_i):
        d = dates[i]
        row: Dict[str, object] = {"date": d.isoformat()}
        breadth = sum(1 for asset in UNIVERSE if prices[asset][i] > sma(prices[asset], SLOW_TREND_WINDOW, i)) / len(UNIVERSE)
        row["breadth_200dma"] = round(breadth, 8)
        tbill_rate = latest_rate_on_or_before(tbill_rates, d)
        row["tbill_3m_rate"] = "" if tbill_rate is None else round(tbill_rate, 8)
        for asset in UNIVERSE:
            row.update({key: round(value, 8) for key, value in price_feature_row(prices, asset, i).items()})
        for asset in UNIVERSE:
            row.update(label_row(dates, prices, asset, i))
        rows.append(row)
    return rows


def main() -> None:
    """生成并覆盖 data/factors/monthly_factors.csv。"""

    rows = build_rows()
    FACTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_csv(FACTOR_PATH, rows, FACTOR_COLUMNS)


if __name__ == "__main__":
    main()
