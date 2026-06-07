"""Prediction module: walk-forward 0-1 upside scores for QQQ, GLD, and SGOV."""
from __future__ import annotations

import csv
import json
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from market_data import (
    FACTOR_DIR,
    FACTOR_PATH,
    FORWARD_RETURN_DAYS,
    MEDIUM_RETURN_WINDOW,
    REPORT_DIR,
    RSI_WINDOW,
    SLOW_TREND_WINDOW,
    UNIVERSE,
    PriceMap,
    monthly_price_indices,
    pct_change,
    rsi,
    save_csv,
    sma,
)

MIN_TRAINING_SAMPLES = 12
RIDGE_LAMBDA = 1e-4
SCORE_RETURN_SCALE = 0.10
SCORE_PROBABILITY_WEIGHT = 0.50

FACTOR_FEATURES = [
    "eps_revision_score",
    "forward_peg_score",
    "fcf_yield_minus_tbill",
    "ai_profit_conversion_score",
    "top_weight_concentration",
    "breadth_200dma",
    "crowding_score",
]
ASSET_PRICE_FEATURES = [
    "asset_extension_200dma",
    "asset_rsi_14",
    "asset_return_63d",
]
FEATURES = FACTOR_FEATURES + ASSET_PRICE_FEATURES
FACTOR_COLUMNS = ["date"] + FACTOR_FEATURES


@dataclass
class AssetObservation:
    """单个月度预测样本：保存当月可见特征，以及未来30日端点/均价两个后验标签。"""

    d: date
    asset: str
    price_i: int
    features: Dict[str, Optional[float]]
    actual_forward_1m_return: Optional[float]
    actual_forward_1m_avg_return: Optional[float]
    label_end_i: Optional[int]


@dataclass
class AssetPrediction:
    """单个月度预测结果：保存模型输出、真实后验、误差和防穿越审计信息。"""

    d: date
    asset: str
    price_i: int
    label_end_i: Optional[int]
    features: Dict[str, Optional[float]]
    predicted_1m_return: float
    score: float
    actual_forward_1m_return: Optional[float]
    actual_forward_1m_avg_return: Optional[float]
    actual_up: Optional[int]
    actual_avg_up: Optional[int]
    prediction_error: Optional[float]
    model_coefficients: Dict[str, float]
    sample_size: int
    max_train_label_end_i: Optional[int]
    estimator_name: str = "model_walk_forward"


class PredictionEstimator:
    """预测策略接口：不同预估方法实现同一输出格式，方便共用 AUC、校准和报告框架。"""

    name = "base"

    def predict(self, observations: List[AssetObservation]) -> List[AssetPrediction]:
        """根据月度样本生成预测结果；子类负责具体模型或规则逻辑。"""

        raise NotImplementedError


class WalkForwardLinearEstimator(PredictionEstimator):
    """线性模型预估策略：复用 walk-forward 训练流程，避免未来数据进入当月预测。"""

    name = "model_walk_forward"

    def predict(self, observations: List[AssetObservation]) -> List[AssetPrediction]:
        """运行线性模型 walk-forward 预测，并保持原有模型预估代码路径不变。"""

        return run_walk_forward_predictions(observations)


class RuleSignalEstimator(PredictionEstimator):
    """规则预估策略：直接把价格是否偏高、RSI 和 63 日趋势映射成 0-1 分数。"""

    name = "rule_signal"

    def predict(self, observations: List[AssetObservation]) -> List[AssetPrediction]:
        """按固定规则生成预测分数；不训练模型，作为可解释的 QQQ 择时预估策略。"""

        return run_rule_signal_predictions(observations)


def ensure_factor_template() -> None:
    """确保月度因子 CSV 模板存在；不存在时只写表头，不填任何未来数据。"""

    if FACTOR_PATH.exists():
        return
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    with FACTOR_PATH.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FACTOR_COLUMNS).writeheader()


def parse_optional_float(value: object) -> Optional[float]:
    """把 CSV 单元格解析为可选浮点数；空值或非法值返回 None，后续由训练均值填充。"""

    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_factor_file() -> Dict[Tuple[int, int], Dict[str, Optional[float]]]:
    """读取月度因子文件，按年月索引，供每个月第一个交易日取当月可见因子。"""

    ensure_factor_template()
    out: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}
    with FACTOR_PATH.open() as f:
        for row in csv.DictReader(f):
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            out[(d.year, d.month)] = {name: parse_optional_float(row.get(name)) for name in FACTOR_FEATURES}
    return out


def asset_price_features(prices: PriceMap, asset: str, i: int) -> Dict[str, float]:
    """计算资产自身价格类兜底因子：200日偏离、14日 RSI、63日涨幅。"""

    series = prices[asset]
    return {
        "asset_extension_200dma": series[i] / sma(series, SLOW_TREND_WINDOW, i) - 1.0,
        "asset_rsi_14": rsi(series, RSI_WINDOW, i),
        "asset_return_63d": pct_change(series, MEDIUM_RETURN_WINDOW, i),
    }


def forward_label_end_index(dates: List[date], i: int) -> Optional[int]:
    """把未来30个自然日映射到实际交易日索引；若目标日休市，则取之后第一个交易日。"""

    target_date = dates[i] + timedelta(days=FORWARD_RETURN_DAYS)
    label_end_i = bisect_left(dates, target_date, i + 1)
    return label_end_i if label_end_i < len(dates) else None


def build_asset_observations(dates: List[date], prices: PriceMap, start_i: int) -> List[AssetObservation]:
    """构造月度训练/预测样本；每个资产每月一行，并生成未来30个自然日端点和均价后验标签。"""

    factor_by_month = load_factor_file()
    observations = []
    for i in monthly_price_indices(dates, start_i):
        d = dates[i]
        base_features = {name: None for name in FACTOR_FEATURES}
        base_features.update(factor_by_month.get((d.year, d.month), {}))
        for asset in UNIVERSE:
            features = dict(base_features)
            features.update(asset_price_features(prices, asset, i))
            label = None
            avg_label = None
            label_end_i = forward_label_end_index(dates, i)
            if label_end_i is not None:
                label = prices[asset][label_end_i] / prices[asset][i] - 1.0
                forward_window = prices[asset][i + 1 : label_end_i + 1]
                avg_label = statistics.mean(forward_window) / prices[asset][i] - 1.0
            observations.append(AssetObservation(d, asset, i, features, label, avg_label, label_end_i))
    return observations


def feature_defaults(train_rows: List[AssetObservation]) -> Dict[str, float]:
    """为缺失因子计算训练集均值；没有历史值的特征使用 0，避免引入未来填充值。"""

    defaults = {}
    for name in FEATURES:
        vals = [row.features.get(name) for row in train_rows if row.features.get(name) is not None]
        defaults[name] = statistics.mean(vals) if vals else 0.0
    return defaults


def feature_scales(train_rows: List[AssetObservation], defaults: Dict[str, float]) -> Dict[str, float]:
    """计算训练集特征标准差，用于标准化；方差过小时用 1 防止除零。"""

    scales = {}
    for name in FEATURES:
        vals = [row.features.get(name, defaults[name]) or defaults[name] for row in train_rows]
        scale = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        scales[name] = scale if scale > 1e-9 else 1.0
    return scales


def vectorize(features: Dict[str, Optional[float]], defaults: Dict[str, float], scales: Dict[str, float]) -> List[float]:
    """把特征字典转成线性模型向量；第一项为截距，其他项按训练集均值/方差标准化。"""

    values = [1.0]
    for name in FEATURES:
        raw = features.get(name)
        value = defaults[name] if raw is None else raw
        values.append((value - defaults[name]) / scales[name])
    return values


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """用高斯消元解正规方程；避免引入 sklearn 依赖。"""

    n = len(b)
    matrix = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            continue
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        pivot_value = matrix[col][col]
        matrix[col] = [v / pivot_value for v in matrix[col]]
        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            matrix[row] = [v - factor * pv for v, pv in zip(matrix[row], matrix[col])]
    return [matrix[i][-1] for i in range(n)]


def fit_asset_model(train_rows: List[AssetObservation], target: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], int]:
    """基于过去已完成后验标签的样本训练线性模型；target 控制训练收益或上涨概率。"""

    usable_rows = [row for row in train_rows if row.actual_forward_1m_return is not None]
    sample_size = len(usable_rows)
    defaults = feature_defaults(usable_rows)
    scales = feature_scales(usable_rows, defaults)
    if sample_size < MIN_TRAINING_SAMPLES:
        intercept = 0.5 if target == "up_probability" else 0.0
        return {"intercept": intercept}, defaults, scales, sample_size

    vectors = [vectorize(row.features, defaults, scales) for row in usable_rows]
    if target == "up_probability":
        labels = [1.0 if (row.actual_forward_1m_return or 0.0) > 0 else 0.0 for row in usable_rows]
    else:
        labels = [row.actual_forward_1m_return or 0.0 for row in usable_rows]
    dim = len(FEATURES) + 1
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for x, y in zip(vectors, labels):
        for r in range(dim):
            xty[r] += x[r] * y
            for c in range(dim):
                xtx[r][c] += x[r] * x[c]
    for i in range(1, dim):
        xtx[i][i] += RIDGE_LAMBDA
    beta = solve_linear_system(xtx, xty)
    coeffs = {"intercept": beta[0]}
    coeffs.update({name: beta[i + 1] for i, name in enumerate(FEATURES)})
    return coeffs, defaults, scales, sample_size


def predict_return(
    features: Dict[str, Optional[float]],
    coeffs: Dict[str, float],
    defaults: Dict[str, float],
    scales: Dict[str, float],
) -> float:
    """用线性模型系数预测当前特征对应的未来收益或上涨概率原始值。"""

    if len(coeffs) == 1:
        return coeffs.get("intercept", 0.0)
    x = vectorize(features, defaults, scales)
    beta = [coeffs["intercept"]] + [coeffs[name] for name in FEATURES]
    return sum(v * b for v, b in zip(x, beta))


def clamp_score(score: float) -> float:
    """把分数限制在 0 到 1 之间，保证策略模块收到稳定的仓位信号。"""

    return max(0.0, min(1.0, score))


def score_from_return(predicted_return: float, scale: float = SCORE_RETURN_SCALE) -> float:
    """把预测收益映射成 0-1 分数；0.5 表示收益接近 0，正负收益按 scale 线性缩放。"""

    if scale <= 0:
        return 0.5
    return clamp_score(0.5 + predicted_return / (2.0 * scale))


def blended_score(probability_score: float, predicted_return: float) -> float:
    """把方向概率分数和收益幅度分数混合，形成兼顾胜率和赔率的候选 score。"""

    return clamp_score(
        SCORE_PROBABILITY_WEIGHT * probability_score
        + (1.0 - SCORE_PROBABILITY_WEIGHT) * score_from_return(predicted_return)
    )


def rule_signal_score(obs: AssetObservation) -> Tuple[float, float, Dict[str, float]]:
    """把单个资产的价格信号转成规则分数；QQQ 偏低且趋势未坏时给更高分。"""

    extension = obs.features.get("asset_extension_200dma") or 0.0
    rsi_14 = obs.features.get("asset_rsi_14") or 50.0
    return_63d = obs.features.get("asset_return_63d") or 0.0
    if obs.asset == "QQQ":
        score = 0.68 - 0.35 * extension - 0.002 * (rsi_14 - 55.0) + 0.35 * return_63d
        if extension > 0.20 or rsi_14 > 75:
            score -= 0.08
        elif extension < 0.05 and rsi_14 < 65:
            score += 0.03
    elif obs.asset == "GLD":
        score = 0.55 - 0.45 * extension - 0.002 * (rsi_14 - 55.0) + 0.15 * return_63d
        if extension > 0.10 or rsi_14 > 70:
            score -= 0.12
    else:
        score = 0.54 + min(0.10, max(-0.05, return_63d * 3.0))
    score = clamp_score(score)
    predicted_return = (score - 0.5) * SCORE_RETURN_SCALE
    diagnostics = {
        "rule_extension_200dma": extension,
        "rule_rsi_14": rsi_14,
        "rule_return_63d": return_63d,
    }
    return score, predicted_return, diagnostics


def run_rule_signal_predictions(observations: List[AssetObservation]) -> List[AssetPrediction]:
    """运行规则预估策略；每个样本只用当月可见价格信号，不做训练也不使用未来 label。"""

    predictions = []
    for obs in observations:
        score, pred, diagnostics = rule_signal_score(obs)
        actual = obs.actual_forward_1m_return
        actual_avg = obs.actual_forward_1m_avg_return
        actual_up = None if actual is None else int(actual > 0)
        actual_avg_up = None if actual_avg is None else int(actual_avg > 0)
        error = None if actual is None else pred - actual
        predictions.append(AssetPrediction(
            obs.d,
            obs.asset,
            obs.price_i,
            obs.label_end_i,
            obs.features,
            pred,
            score,
            actual,
            actual_avg,
            actual_up,
            actual_avg_up,
            error,
            diagnostics,
            0,
            None,
            RuleSignalEstimator.name,
        ))
    return predictions


def run_prediction_strategy(observations: List[AssetObservation], estimator_name: str) -> List[AssetPrediction]:
    """按名称选择预测策略；模型预估和规则预估共享同一个预测结果数据结构。"""

    estimators: Dict[str, PredictionEstimator] = {
        WalkForwardLinearEstimator.name: WalkForwardLinearEstimator(),
        RuleSignalEstimator.name: RuleSignalEstimator(),
    }
    if estimator_name not in estimators:
        raise ValueError(f"Unknown prediction estimator: {estimator_name}")
    return estimators[estimator_name].predict(observations)


def score_auc_from_pairs(scores: List[float], labels: List[int]) -> Optional[float]:
    """根据分数和 0/1 标签计算 AUC；用于端点价格上涨和均价上涨两个口径。"""

    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def score_orientation(
    train_rows: List[AssetObservation],
    return_coeffs: Dict[str, float],
    return_defaults: Dict[str, float],
    return_scales: Dict[str, float],
    score_coeffs: Dict[str, float],
    score_defaults: Dict[str, float],
    score_scales: Dict[str, float],
) -> Tuple[int, Optional[float]]:
    """用训练集内分数 AUC 判断方向是否明显反向；仅作为诊断/兜底，不直接看未来样本。"""

    usable_rows = [row for row in train_rows if row.actual_forward_1m_return is not None]
    if len(usable_rows) < MIN_TRAINING_SAMPLES:
        return 1, None
    scores = []
    labels = []
    for row in usable_rows:
        pred = predict_return(row.features, return_coeffs, return_defaults, return_scales)
        probability_score = clamp_score(predict_return(row.features, score_coeffs, score_defaults, score_scales))
        scores.append(blended_score(probability_score, pred))
        labels.append(int((row.actual_forward_1m_return or 0.0) > 0))
    train_auc = score_auc_from_pairs(scores, labels)
    if train_auc is None:
        return 1, None
    return (-1 if train_auc < 0.5 else 1), train_auc


def oos_score_orientation(predictions: List[AssetPrediction], obs: AssetObservation) -> Tuple[Optional[int], Optional[float], int]:
    """用历史 OOS 预测表现判断分数方向；只使用当前月之前已完成后验的预测记录。"""

    usable_rows = [
        row for row in predictions
        if row.asset == obs.asset
        and row.actual_up is not None
        and row.label_end_i is not None
        and row.label_end_i < obs.price_i
    ]
    if len(usable_rows) < MIN_TRAINING_SAMPLES:
        return None, None, len(usable_rows)
    raw_scores = [
        row.model_coefficients.get("score_raw", row.score)
        for row in usable_rows
    ]
    oos_auc = score_auc_from_pairs(
        raw_scores,
        [row.actual_up or 0 for row in usable_rows],
    )
    if oos_auc is None:
        return None, None, len(usable_rows)
    ordered = sorted(zip(raw_scores, usable_rows), key=lambda item: item[0])
    bucket_size = max(1, len(ordered) // 5)
    low_rows = [row for _, row in ordered[:bucket_size]]
    high_rows = [row for _, row in ordered[-bucket_size:]]
    low_return = statistics.mean(row.actual_forward_1m_return or 0.0 for row in low_rows)
    high_return = statistics.mean(row.actual_forward_1m_return or 0.0 for row in high_rows)
    low_up_rate = statistics.mean(row.actual_up or 0 for row in low_rows)
    high_up_rate = statistics.mean(row.actual_up or 0 for row in high_rows)
    should_flip = oos_auc < 0.48 and high_return < low_return and high_up_rate < low_up_rate
    return (-1 if should_flip else 1), oos_auc, len(usable_rows)


def oos_component_selection(predictions: List[AssetPrediction], obs: AssetObservation) -> Tuple[str, Optional[float], int]:
    """在 raw/probability/return 三种候选分数里，按历史 OOS AUC 选择当前采用的 score 组件。"""

    usable_rows = [
        row for row in predictions
        if row.asset == obs.asset
        and row.actual_up is not None
        and row.label_end_i is not None
        and row.label_end_i < obs.price_i
    ]
    if len(usable_rows) < MIN_TRAINING_SAMPLES:
        return "raw", None, len(usable_rows)
    candidates = {
        "raw": [
            row.model_coefficients.get("score_raw", row.score)
            for row in usable_rows
        ],
        "probability": [
            row.model_coefficients.get("score_probability_component", row.score)
            for row in usable_rows
        ],
        "return": [
            row.model_coefficients.get("score_return_component", row.score)
            for row in usable_rows
        ],
    }
    labels = [row.actual_up or 0 for row in usable_rows]
    aucs = {
        name: score_auc_from_pairs(scores, labels)
        for name, scores in candidates.items()
    }
    available = {name: value for name, value in aucs.items() if value is not None}
    if not available:
        return "raw", None, len(usable_rows)
    best_name = max(available, key=lambda name: available[name] or 0.0)
    return best_name, available[best_name], len(usable_rows)


def run_walk_forward_predictions(observations: List[AssetObservation]) -> List[AssetPrediction]:
    """按时间顺序 walk-forward 预测；每个月只用此前已结束后验标签的样本训练和选分数组件。"""

    predictions = []
    for idx, obs in enumerate(observations):
        train_rows = [
            row for row in observations[:idx]
            if row.asset == obs.asset and row.label_end_i is not None and row.label_end_i < obs.price_i
        ]
        return_coeffs, return_defaults, return_scales, sample_size = fit_asset_model(train_rows, "return")
        score_coeffs, score_defaults, score_scales, _ = fit_asset_model(train_rows, "up_probability")
        pred = predict_return(obs.features, return_coeffs, return_defaults, return_scales)
        probability_score = clamp_score(predict_return(obs.features, score_coeffs, score_defaults, score_scales))
        raw_score = blended_score(probability_score, pred)
        train_orientation, train_score_auc = score_orientation(
            train_rows,
            return_coeffs,
            return_defaults,
            return_scales,
            score_coeffs,
            score_defaults,
            score_scales,
        )
        oos_orientation, oos_score_auc, oos_orientation_samples = oos_score_orientation(predictions, obs)
        orientation = oos_orientation if oos_orientation is not None else train_orientation
        component_scores = {
            "raw": raw_score,
            "probability": probability_score,
            "return": score_from_return(pred),
        }
        selected_component, selected_component_auc, selected_component_samples = oos_component_selection(predictions, obs)
        score = component_scores[selected_component]
        actual = obs.actual_forward_1m_return
        actual_avg = obs.actual_forward_1m_avg_return
        actual_up = None if actual is None else int(actual > 0)
        actual_avg_up = None if actual_avg is None else int(actual_avg > 0)
        error = None if actual is None else pred - actual
        max_train_label_end_i = max((row.label_end_i for row in train_rows if row.label_end_i is not None), default=None)
        coeffs = {f"return_{key}": value for key, value in return_coeffs.items()}
        coeffs.update({f"score_{key}": value for key, value in score_coeffs.items()})
        coeffs["score_probability_component"] = probability_score
        coeffs["score_return_component"] = score_from_return(pred)
        coeffs["score_raw"] = raw_score
        coeffs["score_orientation"] = orientation
        coeffs["score_orientation_source"] = 1 if oos_orientation is not None else 0
        coeffs["score_oos_orientation_samples"] = oos_orientation_samples
        coeffs["score_selected_component"] = {"raw": 0, "probability": 1, "return": 2}[selected_component]
        coeffs["score_selected_component_samples"] = selected_component_samples
        if selected_component_auc is not None:
            coeffs["score_selected_component_auc"] = selected_component_auc
        if oos_score_auc is not None:
            coeffs["score_oos_auc_for_orientation"] = oos_score_auc
        if train_score_auc is not None:
            coeffs["score_train_auc_for_orientation"] = train_score_auc
        predictions.append(AssetPrediction(
            obs.d,
            obs.asset,
            obs.price_i,
            obs.label_end_i,
            obs.features,
            pred,
            score,
            actual,
            actual_avg,
            actual_up,
            actual_avg_up,
            error,
            coeffs,
            sample_size,
            max_train_label_end_i,
            WalkForwardLinearEstimator.name,
        ))
    return predictions


def auc(rows: List[AssetPrediction]) -> Optional[float]:
    """按未来30个自然日对应交易日价格是否上涨计算 AUC，保留为默认端点价评估口径。"""

    evaluated = [row for row in rows if row.actual_up is not None]
    positives = [row.score for row in evaluated if row.actual_up == 1]
    negatives = [row.score for row in evaluated if row.actual_up == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def avg_price_auc(rows: List[AssetPrediction]) -> Optional[float]:
    """按未来30个自然日窗口内交易日平均价格是否高于当前价格计算 AUC。"""

    evaluated = [row for row in rows if row.actual_avg_up is not None]
    return score_auc_from_pairs(
        [row.score for row in evaluated],
        [row.actual_avg_up or 0 for row in evaluated],
    )


def prediction_metrics(predictions: List[AssetPrediction]) -> Dict[str, object]:
    """汇总预测模块评估指标，包含端点价 AUC、30日均价 AUC、误差、校准和特征覆盖率。"""

    evaluated = [row for row in predictions if row.actual_forward_1m_return is not None]
    errors = [row.prediction_error or 0.0 for row in evaluated]
    metrics_by_asset = {}
    yearly_auc: Dict[str, Dict[str, Optional[float]]] = {}
    yearly_avg_price_auc: Dict[str, Dict[str, Optional[float]]] = {}
    for asset in UNIVERSE:
        asset_rows = [row for row in predictions if row.asset == asset]
        asset_eval = [row for row in asset_rows if row.actual_forward_1m_return is not None]
        direction_hits = [
            (row.score >= 0.5) == bool(row.actual_up)
            for row in asset_eval
        ]
        avg_direction_hits = [
            (row.score >= 0.5) == bool(row.actual_avg_up)
            for row in asset_eval
            if row.actual_avg_up is not None
        ]
        metrics_by_asset[asset] = {
            "sample_count": len(asset_eval),
            "auc": None if auc(asset_rows) is None else round(auc(asset_rows) or 0.0, 4),
            "avg_price_auc": None if avg_price_auc(asset_rows) is None else round(avg_price_auc(asset_rows) or 0.0, 4),
            "direction_accuracy": round(sum(direction_hits) / len(direction_hits), 4) if direction_hits else 0.0,
            "avg_price_direction_accuracy": round(sum(avg_direction_hits) / len(avg_direction_hits), 4)
            if avg_direction_hits else 0.0,
            "mae": round(statistics.mean(abs(row.prediction_error or 0.0) for row in asset_eval), 8)
            if asset_eval else 0.0,
        }
        years = sorted({row.d.year for row in asset_eval})
        yearly_auc[asset] = {
            str(year): None if auc([row for row in asset_rows if row.d.year == year]) is None
            else round(auc([row for row in asset_rows if row.d.year == year]) or 0.0, 4)
            for year in years
        }
        yearly_avg_price_auc[asset] = {
            str(year): None if avg_price_auc([row for row in asset_rows if row.d.year == year]) is None
            else round(avg_price_auc([row for row in asset_rows if row.d.year == year]) or 0.0, 4)
            for year in years
        }
    latest_by_asset = {}
    for asset in UNIVERSE:
        latest = [row for row in predictions if row.asset == asset][-1]
        latest_by_asset[asset] = {
            "date": latest.d.isoformat(),
            "score": round(latest.score, 6),
            "predicted_1m_return": round(latest.predicted_1m_return, 6),
            "actual_forward_1m_return": None
            if latest.actual_forward_1m_return is None
            else round(latest.actual_forward_1m_return, 6),
            "actual_forward_1m_avg_return": None
            if latest.actual_forward_1m_avg_return is None
            else round(latest.actual_forward_1m_avg_return, 6),
            "sample_size": latest.sample_size,
        }
    return {
        "estimator": predictions[0].estimator_name if predictions else "",
        "sample_count": len(evaluated),
        "overall_auc": None if auc(predictions) is None else round(auc(predictions) or 0.0, 4),
        "overall_avg_price_auc": None if avg_price_auc(predictions) is None
        else round(avg_price_auc(predictions) or 0.0, 4),
        "mse": round(statistics.mean(e * e for e in errors), 8) if errors else 0.0,
        "mae": round(statistics.mean(abs(e) for e in errors), 8) if errors else 0.0,
        "by_asset": metrics_by_asset,
        "yearly_auc": yearly_auc,
        "yearly_avg_price_auc": yearly_avg_price_auc,
        "score_calibration": score_calibration_summary(predictions),
        "feature_coverage": feature_coverage(predictions),
        "single_feature_auc": single_feature_auc(predictions),
        "latest_predictions": latest_by_asset,
    }


def score_calibration_rows(predictions: List[AssetPrediction], buckets: int = 5) -> List[Dict[str, object]]:
    """按资产内 score 分桶，观察高分桶的端点收益/均价收益是否优于低分桶。"""

    rows = []
    for asset in UNIVERSE:
        evaluated = [
            row for row in predictions
            if row.asset == asset and row.actual_forward_1m_return is not None
        ]
        evaluated.sort(key=lambda row: row.score)
        if not evaluated:
            continue
        for bucket in range(buckets):
            start = bucket * len(evaluated) // buckets
            end = (bucket + 1) * len(evaluated) // buckets
            bucket_rows = evaluated[start:end]
            if not bucket_rows:
                continue
            rows.append({
                "asset": asset,
                "bucket": bucket + 1,
                "bucket_label": "lowest" if bucket == 0 else "highest" if bucket == buckets - 1 else f"{bucket + 1}",
                "sample_count": len(bucket_rows),
                "avg_score": round(statistics.mean(row.score for row in bucket_rows), 6),
                "min_score": round(min(row.score for row in bucket_rows), 6),
                "max_score": round(max(row.score for row in bucket_rows), 6),
                "up_rate": round(statistics.mean(row.actual_up or 0 for row in bucket_rows), 4),
                "avg_price_up_rate": round(statistics.mean(row.actual_avg_up or 0 for row in bucket_rows), 4),
                "avg_actual_forward_1m_return": round(statistics.mean(row.actual_forward_1m_return or 0.0 for row in bucket_rows), 6),
                "avg_actual_forward_1m_avg_return": round(statistics.mean(row.actual_forward_1m_avg_return or 0.0 for row in bucket_rows), 6),
                "avg_predicted_1m_return": round(statistics.mean(row.predicted_1m_return for row in bucket_rows), 6),
            })
    return rows


def score_calibration_summary(predictions: List[AssetPrediction]) -> Dict[str, object]:
    """从分桶校准表提炼高分桶减低分桶的收益差和上涨率差，用于模块评分。"""

    rows = score_calibration_rows(predictions)
    out = {}
    for asset in UNIVERSE:
        asset_rows = [row for row in rows if row["asset"] == asset]
        if len(asset_rows) < 2:
            out[asset] = {
                "available": False,
                "diagnosis": "insufficient_data",
            }
            continue
        low = asset_rows[0]
        high = asset_rows[-1]
        return_spread = high["avg_actual_forward_1m_return"] - low["avg_actual_forward_1m_return"]
        avg_return_spread = high["avg_actual_forward_1m_avg_return"] - low["avg_actual_forward_1m_avg_return"]
        up_rate_spread = high["up_rate"] - low["up_rate"]
        avg_up_rate_spread = high["avg_price_up_rate"] - low["avg_price_up_rate"]
        out[asset] = {
            "available": True,
            "lowest_bucket_avg_return": low["avg_actual_forward_1m_return"],
            "highest_bucket_avg_return": high["avg_actual_forward_1m_return"],
            "high_minus_low_return": round(return_spread, 6),
            "lowest_bucket_avg_price_return": low["avg_actual_forward_1m_avg_return"],
            "highest_bucket_avg_price_return": high["avg_actual_forward_1m_avg_return"],
            "high_minus_low_avg_price_return": round(avg_return_spread, 6),
            "lowest_bucket_up_rate": low["up_rate"],
            "highest_bucket_up_rate": high["up_rate"],
            "high_minus_low_up_rate": round(up_rate_spread, 4),
            "lowest_bucket_avg_price_up_rate": low["avg_price_up_rate"],
            "highest_bucket_avg_price_up_rate": high["avg_price_up_rate"],
            "high_minus_low_avg_price_up_rate": round(avg_up_rate_spread, 4),
            "monotonic_return_buckets": all(
                asset_rows[i]["avg_actual_forward_1m_return"] <= asset_rows[i + 1]["avg_actual_forward_1m_return"]
                for i in range(len(asset_rows) - 1)
            ),
            "monotonic_avg_price_return_buckets": all(
                asset_rows[i]["avg_actual_forward_1m_avg_return"] <= asset_rows[i + 1]["avg_actual_forward_1m_avg_return"]
                for i in range(len(asset_rows) - 1)
            ),
        }
    return out


def feature_coverage(predictions: List[AssetPrediction]) -> Dict[str, object]:
    """统计每个特征在可评估样本中的非空覆盖率，用于判断基本面因子是否足够训练。"""

    evaluated = [row for row in predictions if row.actual_up is not None]
    out = {}
    for name in FEATURES:
        present = sum(1 for row in evaluated if row.features.get(name) is not None)
        out[name] = {
            "present_count": present,
            "total_count": len(evaluated),
            "coverage": round(present / len(evaluated), 4) if evaluated else 0.0,
        }
    return out


def single_feature_auc(predictions: List[AssetPrediction]) -> Dict[str, object]:
    """逐个特征单独作为分数计算 AUC，粗略判断哪些因子本身有排序能力。"""

    out = {}
    evaluated = [row for row in predictions if row.actual_up is not None]
    for name in FEATURES:
        rows = []
        values = [row.features.get(name) for row in evaluated if row.features.get(name) is not None]
        if len(values) < 10:
            out[name] = {"auc": None, "coverage": 0.0, "direction": "insufficient_data"}
            continue
        for row in evaluated:
            value = row.features.get(name)
            if value is None:
                continue
            proxy = AssetPrediction(
                row.d,
                row.asset,
                row.price_i,
                row.label_end_i,
                row.features,
                value,
                value,
                row.actual_forward_1m_return,
                row.actual_forward_1m_avg_return,
                row.actual_up,
                row.actual_avg_up,
                None,
                {},
                row.sample_size,
                row.max_train_label_end_i,
            )
            rows.append(proxy)
        raw_auc = auc(rows)
        if raw_auc is None:
            out[name] = {"auc": None, "coverage": round(len(rows) / len(evaluated), 4), "direction": "unavailable"}
            continue
        flipped = 1.0 - raw_auc
        if flipped > raw_auc:
            out[name] = {
                "auc": round(flipped, 4),
                "coverage": round(len(rows) / len(evaluated), 4),
                "direction": "lower_is_better",
            }
        else:
            out[name] = {
                "auc": round(raw_auc, 4),
                "coverage": round(len(rows) / len(evaluated), 4),
                "direction": "higher_is_better",
            }
    return out


def save_prediction_outputs(predictions: List[AssetPrediction], report_prefix: str = "") -> Dict[str, object]:
    """写出预测明细、校准表和预测指标 JSON；prefix 用于区分规则预估和模型预估。"""

    rows = []
    for row in predictions:
        rows.append({
            "date": row.d.isoformat(),
            "asset": row.asset,
            "estimator": row.estimator_name,
            **{name: "" if row.features.get(name) is None else round(row.features[name] or 0.0, 8) for name in FEATURES},
            "score": round(row.score, 8),
            "predicted_1m_return": round(row.predicted_1m_return, 8),
            "actual_forward_1m_return": ""
            if row.actual_forward_1m_return is None
            else round(row.actual_forward_1m_return, 8),
            "actual_forward_1m_avg_return": ""
            if row.actual_forward_1m_avg_return is None
            else round(row.actual_forward_1m_avg_return, 8),
            "label_end_i": "" if row.label_end_i is None else row.label_end_i,
            "max_train_label_end_i": "" if row.max_train_label_end_i is None else row.max_train_label_end_i,
            "actual_up": "" if row.actual_up is None else row.actual_up,
            "actual_avg_up": "" if row.actual_avg_up is None else row.actual_avg_up,
            "prediction_error": "" if row.prediction_error is None else round(row.prediction_error, 8),
            "model_coefficients": json.dumps({k: round(v, 8) for k, v in row.model_coefficients.items()}, sort_keys=True),
            "sample_size": row.sample_size,
        })
    name_prefix = f"{report_prefix}_" if report_prefix else ""
    save_csv(REPORT_DIR / f"{name_prefix}factor_predictions.csv", rows, list(rows[0].keys()))
    calibration_rows = score_calibration_rows(predictions)
    if calibration_rows:
        save_csv(REPORT_DIR / f"{name_prefix}prediction_calibration.csv", calibration_rows, list(calibration_rows[0].keys()))
    metrics = prediction_metrics(predictions)
    with (REPORT_DIR / f"{name_prefix}prediction_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def leakage_audit(predictions: List[AssetPrediction]) -> Dict[str, object]:
    """检查 walk-forward 是否穿越：训练标签结束日必须早于当前预测日。"""

    violations = []
    warmup_rows = 0
    for row in predictions:
        if row.sample_size < MIN_TRAINING_SAMPLES:
            warmup_rows += 1
        if row.max_train_label_end_i is not None and row.max_train_label_end_i >= row.price_i:
            violations.append({
                "date": row.d.isoformat(),
                "asset": row.asset,
                "price_i": row.price_i,
                "max_train_label_end_i": row.max_train_label_end_i,
            })
    audit = {
        "passed": len(violations) == 0,
        "checked_predictions": len(predictions),
        "violations": violations,
        "warmup_predictions_with_neutral_model": warmup_rows,
        "rule": (
            "A training observation is allowed only when its forward-label end index is strictly "
            "less than the current prediction index."
        ),
    }
    with (REPORT_DIR / "leakage_audit.json").open("w") as f:
        json.dump(audit, f, indent=2)
    return audit


def predictions_by_month(predictions: List[AssetPrediction]) -> Dict[Tuple[int, int], Dict[str, AssetPrediction]]:
    """把预测结果按年月聚合，方便策略模块在每个月首个交易日取当月信号。"""

    out: Dict[Tuple[int, int], Dict[str, AssetPrediction]] = {}
    for row in predictions:
        out.setdefault((row.d.year, row.d.month), {})[row.asset] = row
    return out
