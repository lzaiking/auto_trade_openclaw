"""Strategy module: consume predicted scores and returns, without training models."""
from __future__ import annotations

import math
import statistics
from datetime import date
from typing import Dict, List, Tuple

from market_data import (
    GOLD_ASSET,
    GROWTH_ASSET,
    MONTHLY_CONTRIBUTION,
    SAFE_ASSET,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE,
    Holdings,
    PriceMap,
    portfolio_value,
    rebalance,
)
from prediction_module import AssetPrediction, predictions_by_month


class AllocationStrategy:
    """交易策略接口：不同仓位规则共用同一回测框架，便于增加新的配置方法。"""

    name = "base"

    def weights(self, predictions: Dict[str, AssetPrediction], **config: float) -> Dict[str, float]:
        """根据当月预测信号生成 QQQ/GLD/SGOV 目标权重。"""

        raise NotImplementedError


class RuleSignalAllocationStrategy(AllocationStrategy):
    """规则版 QQQ 择时策略：以定投 QQQ 为核心，只有偏高或过热时才转入防守资产。"""

    name = "rule_signal_strategy"

    def weights(self, predictions: Dict[str, AssetPrediction], **config: float) -> Dict[str, float]:
        """调用规则权重函数，恢复旧版直接看 QQQ 偏离、RSI 和 63 日趋势的交易逻辑。"""

        return rule_signal_weights(predictions, **config)


class LegacySignalAllocationStrategy(AllocationStrategy):
    """截图版信号策略：只消费预估模块分数，把 QQQ score 映射为四档核心/防守仓位。"""

    name = "legacy_signal_strategy"

    def weights(self, predictions: Dict[str, AssetPrediction], **config: float) -> Dict[str, float]:
        """调用截图版仓位映射；预测模块负责打分，策略模块只做固定仓位决策。"""

        return legacy_signal_weights(predictions, **config)


class ScoreReturnAllocationStrategy(AllocationStrategy):
    """模型分数策略：使用预测收益和 0-1 score 做多资产吸引力分配，作为模型对照。"""

    name = "score_return_strategy"

    def weights(self, predictions: Dict[str, AssetPrediction], **config: float) -> Dict[str, float]:
        """调用模型分数权重函数，保留原来的预测驱动配置策略。"""

        return score_return_weights(predictions, **config)


class FixedThresholdAllocationStrategy(AllocationStrategy):
    """固定阈值策略：只根据 QQQ 预测收益档位做简单防守，作为传统对照。"""

    name = "fixed_threshold_strategy"

    def weights(self, predictions: Dict[str, AssetPrediction], **config: float) -> Dict[str, float]:
        """调用固定阈值权重函数。"""

        return fixed_threshold_weights(predictions, **config)


def is_gld_expensive(gld_prediction: AssetPrediction) -> bool:
    """判断 GLD 是否偏高：价格偏离 200 日均线过多或 RSI 过热时，不再优先买入 GLD。"""

    extension = gld_prediction.features.get("asset_extension_200dma")
    rsi_14 = gld_prediction.features.get("asset_rsi_14")
    return (extension is not None and extension >= 0.10) or (rsi_14 is not None and rsi_14 >= 70.0)


def fixed_threshold_weights(predictions: Dict[str, AssetPrediction], **_: float) -> Dict[str, float]:
    """固定阈值对照策略：只看 QQQ 预测收益，把防守仓位分给 GLD 或 SGOV。"""

    qqq_return = predictions[GROWTH_ASSET].predicted_1m_return
    if qqq_return >= 0.03:
        qqq_weight, defensive_weight = 1.0, 0.0
    elif qqq_return >= 0.01:
        qqq_weight, defensive_weight = 0.90, 0.10
    elif qqq_return >= -0.01:
        qqq_weight, defensive_weight = 0.70, 0.30
    else:
        qqq_weight, defensive_weight = 0.40, 0.60

    defensive_asset = SAFE_ASSET if is_gld_expensive(predictions[GOLD_ASSET]) else GOLD_ASSET
    return {
        GROWTH_ASSET: qqq_weight,
        GOLD_ASSET: defensive_weight if defensive_asset == GOLD_ASSET else 0.0,
        SAFE_ASSET: defensive_weight if defensive_asset == SAFE_ASSET else 0.0,
    }


def rule_signal_weights(
    predictions: Dict[str, AssetPrediction],
    mild_defensive_weight: float = 0.02,
    medium_defensive_weight: float = 0.12,
    hot_defensive_weight: float = 0.15,
    severe_defensive_weight: float = 0.20,
) -> Dict[str, float]:
    """旧逻辑主策略：直接用 QQQ 偏离 200 日均线、RSI 和 63 日趋势决定防守比例。"""

    qqq = predictions[GROWTH_ASSET]
    qqq_extension = qqq.features.get("asset_extension_200dma") or 0.0
    qqq_rsi = qqq.features.get("asset_rsi_14") or 50.0
    qqq_return_63d = qqq.features.get("asset_return_63d") or 0.0

    defensive_weight = 0.0
    if qqq_extension >= 0.24 or qqq_rsi >= 78:
        defensive_weight = severe_defensive_weight
    elif qqq_extension >= 0.18 or qqq_rsi >= 72:
        defensive_weight = hot_defensive_weight
    elif qqq_extension >= 0.12 or (qqq_rsi >= 67 and qqq_return_63d > 0.08):
        defensive_weight = medium_defensive_weight
    elif qqq_extension >= 0.06 or qqq_rsi >= 63:
        defensive_weight = mild_defensive_weight

    if qqq_return_63d < -0.10 and qqq_extension < 0.08:
        defensive_weight = min(defensive_weight, mild_defensive_weight)

    defensive_asset = SAFE_ASSET if is_gld_expensive(predictions[GOLD_ASSET]) else GOLD_ASSET
    return {
        GROWTH_ASSET: 1.0 - defensive_weight,
        GOLD_ASSET: defensive_weight if defensive_asset == GOLD_ASSET else 0.0,
        SAFE_ASSET: defensive_weight if defensive_asset == SAFE_ASSET else 0.0,
    }


def legacy_signal_weights(predictions: Dict[str, AssetPrediction], **_: float) -> Dict[str, float]:
    """截图版仓位规则：按 QQQ score 四档配置 QQQ，防守仓位优先 GLD，否则转入 SGOV。"""

    qqq_score = predictions[GROWTH_ASSET].score
    gld_prediction = predictions[GOLD_ASSET]
    gld_score = gld_prediction.score

    if qqq_score >= 0.60:
        qqq_weight, defensive_weight = 1.0, 0.0
    elif qqq_score >= 0.52:
        qqq_weight, defensive_weight = 0.90, 0.10
    elif qqq_score >= 0.45:
        qqq_weight, defensive_weight = 0.75, 0.25
    else:
        qqq_weight, defensive_weight = 0.50, 0.50

    defensive_asset = SAFE_ASSET if gld_score < 0.50 or is_gld_expensive(gld_prediction) else GOLD_ASSET
    return {
        GROWTH_ASSET: qqq_weight,
        GOLD_ASSET: defensive_weight if defensive_asset == GOLD_ASSET else 0.0,
        SAFE_ASSET: defensive_weight if defensive_asset == SAFE_ASSET else 0.0,
    }


def score_return_attractiveness(prediction: AssetPrediction, return_scale: float) -> float:
    """把 0-1 分数优势和正向预测收益合成资产吸引力，用于多资产权重分配。"""

    score_edge = max(0.0, prediction.score - 0.5)
    return_edge = max(0.0, prediction.predicted_1m_return) / return_scale
    return score_edge + return_edge


def score_return_weights(
    predictions: Dict[str, AssetPrediction],
    qqq_min_weight: float = 0.85,
    max_gld_weight: float = 0.15,
    max_sgov_weight: float = 0.20,
    return_scale: float = 0.05,
    neutral_cash_bias: float = 0.02,
) -> Dict[str, float]:
    """正式策略权重：保留 QQQ 核心仓位，再按 QQQ/GLD/SGOV 的吸引力分配剩余仓位。"""

    qqq = predictions[GROWTH_ASSET]
    raw = {
        asset: score_return_attractiveness(pred, return_scale)
        for asset, pred in predictions.items()
    }
    if is_gld_expensive(predictions[GOLD_ASSET]):
        raw[GOLD_ASSET] = 0.0

    if sum(raw.values()) <= 0:
        raw = {GROWTH_ASSET: neutral_cash_bias, GOLD_ASSET: 0.0, SAFE_ASSET: neutral_cash_bias}

    total = sum(raw.values())
    variable_weights = {asset: value / total for asset, value in raw.items()}
    defensive_budget = 1.0 - qqq_min_weight
    weights = {
        GROWTH_ASSET: qqq_min_weight + defensive_budget * variable_weights[GROWTH_ASSET],
        GOLD_ASSET: defensive_budget * variable_weights[GOLD_ASSET],
        SAFE_ASSET: defensive_budget * variable_weights[SAFE_ASSET],
    }

    if weights[GOLD_ASSET] > max_gld_weight:
        excess = weights[GOLD_ASSET] - max_gld_weight
        weights[GOLD_ASSET] = max_gld_weight
        weights[SAFE_ASSET] += excess
    if weights[SAFE_ASSET] > max_sgov_weight:
        excess = weights[SAFE_ASSET] - max_sgov_weight
        weights[SAFE_ASSET] = max_sgov_weight
        weights[GROWTH_ASSET] += excess

    if qqq.score < 0.45 and qqq.predicted_1m_return < -0.01:
        risk_off_shift = min(0.20, weights[GROWTH_ASSET] - 0.40)
        if risk_off_shift > 0:
            weights[GROWTH_ASSET] -= risk_off_shift
            weights[SAFE_ASSET] += risk_off_shift

    total = sum(weights.values())
    return {asset: weights[asset] / total for asset in UNIVERSE}


def target_weights_from_predictions(
    predictions: Dict[str, AssetPrediction],
    method: str = "legacy_signal_strategy",
    **config: float,
) -> Dict[str, float]:
    """根据策略名称选择仓位策略；策略模块只消费预测结果，不参与模型训练。"""

    strategies: Dict[str, AllocationStrategy] = {
        LegacySignalAllocationStrategy.name: LegacySignalAllocationStrategy(),
        RuleSignalAllocationStrategy.name: RuleSignalAllocationStrategy(),
        ScoreReturnAllocationStrategy.name: ScoreReturnAllocationStrategy(),
        FixedThresholdAllocationStrategy.name: FixedThresholdAllocationStrategy(),
    }
    if method not in strategies:
        raise ValueError(f"Unknown allocation strategy: {method}")
    return strategies[method].weights(predictions, **config)


def oracle_weights(predictions: Dict[str, AssetPrediction]) -> Dict[str, float]:
    """理想上限场景：事后选择未来30日实际收益最高的资产，用于估算理论空间。"""

    actuals = {
        asset: pred.actual_forward_1m_return
        for asset, pred in predictions.items()
        if pred.actual_forward_1m_return is not None
    }
    if not actuals:
        return {GROWTH_ASSET: 1.0, GOLD_ASSET: 0.0, SAFE_ASSET: 0.0}
    best = max(actuals, key=lambda asset: actuals[asset] or 0.0)
    return {asset: 1.0 if asset == best else 0.0 for asset in UNIVERSE}


def worst_case_weights(predictions: Dict[str, AssetPrediction]) -> Dict[str, float]:
    """最坏场景：事后选择未来30日实际收益最低的资产，用于估算错误信号风险。"""

    actuals = {
        asset: pred.actual_forward_1m_return
        for asset, pred in predictions.items()
        if pred.actual_forward_1m_return is not None
    }
    if not actuals:
        return {GROWTH_ASSET: 1.0, GOLD_ASSET: 0.0, SAFE_ASSET: 0.0}
    worst = min(actuals, key=lambda asset: actuals[asset] or 0.0)
    return {asset: 1.0 if asset == worst else 0.0 for asset in UNIVERSE}


def drawdown_curve(equities: List[float]) -> List[float]:
    """根据权益曲线计算逐日回撤序列。"""

    peak = equities[0]
    drawdowns = []
    for equity in equities:
        peak = max(peak, equity)
        drawdowns.append(0.0 if peak <= 0 else 1.0 - equity / peak)
    return drawdowns


def xirr(cashflows: List[Tuple[date, float]]) -> float:
    """用二分法计算不规则现金流年化 IRR；月供为负现金流，期末权益为正现金流。"""

    if not cashflows or not any(amount < 0 for _, amount in cashflows) or not any(amount > 0 for _, amount in cashflows):
        return 0.0
    start = cashflows[0][0]

    def npv(rate: float) -> float:
        """按给定折现率计算现金流净现值，供 IRR 二分搜索使用。"""

        return sum(amount / ((1.0 + rate) ** ((d - start).days / 365.25)) for d, amount in cashflows)

    lo, hi = -0.95, 5.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def performance_metrics(equities: List[float], cashflows: List[Tuple[date, float]], end_date: date) -> Dict[str, float]:
    """计算组合核心指标：本金、期末权益、本金收益率、IRR、波动率和 Sharpe。"""

    daily_returns = [equities[i] / equities[i - 1] - 1.0 for i in range(1, len(equities)) if equities[i - 1] > 0]
    total_contributed = -sum(amount for _, amount in cashflows if amount < 0)
    daily_vol = statistics.pstdev(daily_returns) if daily_returns else 0.0
    sharpe = 0.0
    if daily_vol > 0:
        sharpe = statistics.mean(daily_returns) / daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "total_principal": total_contributed,
        "end_equity": equities[-1],
        "total_return_on_principal": equities[-1] / total_contributed - 1.0 if total_contributed else 0.0,
        "annualized_irr": xirr(cashflows + [(end_date, equities[-1])]),
        "annualized_volatility": daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe": sharpe,
    }


def rounded_summary(metrics: Dict[str, float], max_drawdown: float, extra: Dict[str, object]) -> Dict[str, object]:
    """把绩效指标四舍五入并补充策略元信息，作为报告摘要输出。"""

    return {
        **extra,
        "monthly_contribution": MONTHLY_CONTRIBUTION,
        "total_principal": round(metrics["total_principal"], 2),
        "end_equity": round(metrics["end_equity"], 2),
        "total_return_on_principal": round(metrics["total_return_on_principal"], 4),
        "annualized_irr": round(metrics["annualized_irr"], 4),
        "annualized_volatility": round(metrics["annualized_volatility"], 4),
        "sharpe": round(metrics["sharpe"], 4),
        "max_drawdown": round(max_drawdown, 4),
    }


def run_monthly_qqq_dca(dates: List[date], prices: PriceMap, start_i: int) -> Dict[str, object]:
    """基线策略：每月第一个交易日投入 10000 并全部买 QQQ。"""

    shares = 0.0
    contributed_months = set()
    cashflows: List[Tuple[date, float]] = []
    curve = []
    for i in range(start_i, len(dates)):
        d = dates[i]
        month_key = (d.year, d.month)
        if month_key not in contributed_months:
            shares += MONTHLY_CONTRIBUTION / prices[GROWTH_ASSET][i]
            cashflows.append((d, -MONTHLY_CONTRIBUTION))
            contributed_months.add(month_key)
        curve.append({"date": d.isoformat(), "dca_equity": round(shares * prices[GROWTH_ASSET][i], 2)})
    equities = [row["dca_equity"] for row in curve]
    drawdowns = drawdown_curve(equities)
    for row, dd in zip(curve, drawdowns):
        row["dca_drawdown"] = round(dd, 6)
    metrics = performance_metrics(equities, cashflows, dates[-1])
    return {
        "curve": curve,
        "summary": rounded_summary(metrics, max(drawdowns), {"symbol": GROWTH_ASSET, "method": "monthly_qqq_dca"}),
    }


def run_allocated_strategy(
    dates: List[date],
    prices: PriceMap,
    predictions: List[AssetPrediction],
    method: str,
    strategy_config: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """执行月度调仓策略：每月加入新资金，根据当前月预测信号计算目标权重并再平衡。"""

    monthly_predictions = predictions_by_month(predictions)
    holdings: Holdings = {sym: 0.0 for sym in UNIVERSE}
    cashflows: List[Tuple[date, float]] = []
    curve = []
    latest_weights = {sym: 0.0 for sym in UNIVERSE}
    latest_predictions = monthly_predictions[(predictions[0].d.year, predictions[0].d.month)]

    for i in range(predictions[0].price_i, len(dates)):
        d = dates[i]
        month_predictions = monthly_predictions.get((d.year, d.month))
        if month_predictions and d == next(iter(month_predictions.values())).d:
            equity = portfolio_value(holdings, prices, i) + MONTHLY_CONTRIBUTION
            cashflows.append((d, -MONTHLY_CONTRIBUTION))
            latest_predictions = month_predictions
            if method == "oracle":
                latest_weights = oracle_weights(month_predictions)
            elif method == "worst_case":
                latest_weights = worst_case_weights(month_predictions)
            else:
                latest_weights = target_weights_from_predictions(month_predictions, method, **(strategy_config or {}))
            holdings = rebalance(holdings, prices, i, equity, latest_weights)

        row = {
            "date": d.isoformat(),
            "equity": round(portfolio_value(holdings, prices, i), 2),
            "predicted_1m_return": round(latest_predictions[GROWTH_ASSET].predicted_1m_return, 8),
            "actual_forward_1m_return": (
                "" if latest_predictions[GROWTH_ASSET].actual_forward_1m_return is None
                else round(latest_predictions[GROWTH_ASSET].actual_forward_1m_return or 0.0, 8)
            ),
            **{f"score_{asset}": round(latest_predictions[asset].score, 6) for asset in UNIVERSE},
            **{f"pred_{asset}": round(latest_predictions[asset].predicted_1m_return, 8) for asset in UNIVERSE},
            **{f"actual_fwd_{asset}": "" if latest_predictions[asset].actual_forward_1m_return is None
               else round(latest_predictions[asset].actual_forward_1m_return or 0.0, 8) for asset in UNIVERSE},
            **{f"w_{asset}": round(latest_weights.get(asset, 0.0), 4) for asset in UNIVERSE},
        }
        curve.append(row)

    equities = [row["equity"] for row in curve]
    drawdowns = drawdown_curve(equities)
    for row, dd in zip(curve, drawdowns):
        row["drawdown"] = round(dd, 6)
    metrics = performance_metrics(equities, cashflows, dates[-1])
    return {
        "curve": curve,
        "summary": rounded_summary(metrics, max(drawdowns), {
            "method": method,
            "strategy_config": strategy_config or {},
            "latest_target_weights": {k: round(v, 4) for k, v in latest_weights.items() if v > 0},
        }),
        "holdings": holdings,
        "latest_weights": latest_weights,
    }


def strategy_sweep(dates: List[date], prices: PriceMap, predictions: List[AssetPrediction], baseline_max_drawdown: float) -> Dict[str, object]:
    """扫描几组保守仓位参数，比较收益和回撤；仅用于策略模块诊断，不训练预测模型。"""

    rows = []
    candidates = [
        ("legacy_signal_strategy", {}),
        ("rule_signal_strategy", {}),
        ("score_return_strategy", {"qqq_min_weight": 0.65, "max_gld_weight": 0.25, "max_sgov_weight": 0.35}),
        ("score_return_strategy", {"qqq_min_weight": 0.75, "max_gld_weight": 0.20, "max_sgov_weight": 0.25}),
        ("score_return_strategy", {"qqq_min_weight": 0.85, "max_gld_weight": 0.15, "max_sgov_weight": 0.20}),
        ("fixed_threshold_strategy", {}),
    ]
    for method, config in candidates:
        result = run_allocated_strategy(dates, prices, predictions, method, config)
        summary = result["summary"]
        rows.append({
            "strategy_rule": method,
            **config,
            "annualized_irr": summary["annualized_irr"],
            "max_drawdown": summary["max_drawdown"],
            "end_equity": summary["end_equity"],
            "drawdown_ok": summary["max_drawdown"] <= baseline_max_drawdown,
        })
    feasible = [row for row in rows if row["drawdown_ok"]]
    best = max(feasible or rows, key=lambda row: row["annualized_irr"])
    return {
        "rows": rows,
        "best": best,
        "feasible_count": len(feasible),
        "total_count": len(rows),
    }


def build_orders(
    prices: PriceMap,
    i: int,
    current_holdings: Holdings,
    equity_after_contribution: float,
    weights: Dict[str, float],
) -> List[Dict[str, object]]:
    """根据当前持仓、最新价格和目标权重，生成下一次月供后的买卖差额建议。"""

    orders = []
    for sym in UNIVERSE:
        px = prices[sym][i]
        target_qty = math.floor(equity_after_contribution * weights.get(sym, 0.0) / px) if px > 0 else 0
        current_qty = math.floor(current_holdings.get(sym, 0.0))
        delta_qty = target_qty - current_qty
        if delta_qty == 0:
            continue
        orders.append({
            "symbol": sym,
            "side": "BUY" if delta_qty > 0 else "SELL",
            "qty": abs(delta_qty),
            "est_price": round(px, 2),
            "est_value": round(abs(delta_qty) * px, 2),
            "target_weight": round(weights.get(sym, 0.0), 4),
        })
    return orders
