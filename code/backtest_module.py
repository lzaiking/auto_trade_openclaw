"""Backtest module: leakage-safe reports for prediction, strategy, and scenarios."""
from __future__ import annotations

import json
from typing import Dict, List

from market_data import (
    FORWARD_RETURN_DAYS,
    MAX_DRAWDOWN_TARGET,
    MONTHLY_CONTRIBUTION,
    REPORT_DIR,
    SLOW_TREND_WINDOW,
    UNIVERSE,
    load_market_data,
    portfolio_value,
    save_csv,
)
from factor_validation import validate_factor_file
from prediction_module import build_asset_observations, leakage_audit, run_prediction_strategy, run_walk_forward_predictions, save_prediction_outputs
from strategy_module import build_orders, run_allocated_strategy, run_monthly_qqq_dca, strategy_sweep

TARGET_ANNUALIZED_IRR = 0.30
TARGET_AUC = 0.60


def derive_risk_controls(strategy: Dict[str, object], oracle: Dict[str, object], worst_case: Dict[str, object], dca: Dict[str, object]) -> Dict[str, object]:
    """根据策略、基线、oracle 和 worst-case 表现生成预警线、降风险线和止损线。"""

    strategy_summary = strategy["summary"]
    dca_summary = dca["summary"]
    oracle_summary = oracle["summary"]
    worst_summary = worst_case["summary"]

    drawdown_stop = min(MAX_DRAWDOWN_TARGET, dca_summary["max_drawdown"])
    warning_line = drawdown_stop * 0.75
    de_risk_line = drawdown_stop * 0.9
    annualized_gap = TARGET_ANNUALIZED_IRR - strategy_summary["annualized_irr"]
    oracle_headroom = oracle_summary["annualized_irr"] - strategy_summary["annualized_irr"]
    wrong_prediction_loss = strategy_summary["end_equity"] - worst_summary["end_equity"]

    return {
        "portfolio_stop_loss_pct": round(drawdown_stop, 4),
        "warning_drawdown_pct": round(warning_line, 4),
        "de_risk_drawdown_pct": round(de_risk_line, 4),
        "rule": (
            "If portfolio drawdown reaches warning line, cap non-QQQ tilt at 5%; "
            "if it reaches de-risk line, keep at least 30% in SGOV; "
            "if it reaches stop line, pause new GLD/QQQ tilts and direct new contribution to SGOV until next monthly review."
        ),
        "annualized_irr_gap_to_30pct": round(annualized_gap, 4),
        "oracle_annualized_headroom": round(oracle_headroom, 4),
        "worst_case_end_equity_loss_vs_strategy": round(wrong_prediction_loss, 2),
        "interpretation": (
            "Oracle headroom shows the asset-rotation opportunity is large, but current AUC is too weak; "
            "risk lines should stay close to the QQQ DCA drawdown until prediction AUC improves."
        ),
    }


def write_module_scorecard(
    prediction_stats: Dict[str, object],
    leakage_stats: Dict[str, object],
    factor_validation: Dict[str, object],
    strategy: Dict[str, object],
    fixed_threshold: Dict[str, object],
    dca: Dict[str, object],
    oracle: Dict[str, object],
    sweep: Dict[str, object],
) -> Dict[str, object]:
    """分别给预测、策略、回测三个模块打分，明确当前瓶颈和是否可放大仓位。"""

    strategy_summary = strategy["summary"]
    dca_summary = dca["summary"]
    oracle_summary = oracle["summary"]
    overall_auc = prediction_stats.get("overall_auc") or 0.0
    overall_avg_price_auc = prediction_stats.get("overall_avg_price_auc") or 0.0
    coverage = prediction_stats.get("feature_coverage", {})
    missing_factor_features = [
        name for name, stats in coverage.items()
        if name in {
            "eps_revision_score",
            "forward_peg_score",
            "fcf_yield_minus_tbill",
            "ai_profit_conversion_score",
            "top_weight_concentration",
            "breadth_200dma",
            "crowding_score",
        } and stats.get("coverage", 0.0) < 0.5
    ]
    annualized_irr = strategy_summary["annualized_irr"]
    max_drawdown = strategy_summary["max_drawdown"]
    dca_drawdown = dca_summary["max_drawdown"]
    factor_stats = factor_validation.get("feature_stats", {})
    factor_coverage_ready = bool(factor_stats) and all(
        stats.get("coverage", 0.0) >= 0.8
        for stats in factor_stats.values()
    )
    calibration = prediction_stats.get("score_calibration", {})
    calibration_ready = bool(calibration) and all(
        stats.get("available")
        and stats.get("high_minus_low_return", 0.0) > 0
        and stats.get("high_minus_low_up_rate", 0.0) > 0
        for stats in calibration.values()
    )
    prediction_passed = (
        overall_auc >= TARGET_AUC
        and leakage_stats["passed"]
        and factor_validation["passed_structure"]
        and factor_coverage_ready
        and calibration_ready
    )
    scorecard = {
        "prediction_module": {
            "overall_auc": overall_auc,
            "overall_avg_price_auc": overall_avg_price_auc,
            "target_auc": TARGET_AUC,
            "passed": prediction_passed,
            "leakage_audit_passed": leakage_stats["passed"],
            "factor_structure_passed": factor_validation["passed_structure"],
            "factor_coverage_ready": factor_coverage_ready,
            "calibration_ready": calibration_ready,
            "missing_factor_features": missing_factor_features,
            "diagnosis": (
                "Prediction module is not strong enough for aggressive allocation; key monthly factor coverage is still missing."
                if not prediction_passed else
                "Prediction module meets the initial AUC threshold."
            ),
        },
        "strategy_module": {
            "annualized_irr": annualized_irr,
            "target_annualized_irr": TARGET_ANNUALIZED_IRR,
            "max_drawdown": max_drawdown,
            "baseline_drawdown": dca_drawdown,
            "fixed_threshold_annualized_irr": fixed_threshold["summary"]["annualized_irr"],
            "fixed_threshold_max_drawdown": fixed_threshold["summary"]["max_drawdown"],
            "best_sweep_annualized_irr": sweep["best"]["annualized_irr"],
            "best_sweep_config": {
                key: value for key, value in sweep["best"].items()
                if key not in {"annualized_irr", "max_drawdown", "end_equity", "drawdown_ok"}
            },
            "passed": annualized_irr >= TARGET_ANNUALIZED_IRR and max_drawdown <= dca_drawdown,
            "diagnosis": (
                "Strategy should remain conservative until prediction AUC improves."
                if not prediction_passed else
                "Strategy can be tested with larger tilts because prediction AUC is acceptable."
            ),
        },
        "backtest_module": {
            "oracle_annualized_irr": oracle_summary["annualized_irr"],
            "oracle_headroom_to_strategy": round(oracle_summary["annualized_irr"] - annualized_irr, 4),
            "passed": leakage_stats["passed"],
            "diagnosis": "Backtest is leakage-checked and reports oracle/worst-case scenarios.",
        },
        "goal_status": {
            "meets_30pct_annualized_target": annualized_irr >= TARGET_ANNUALIZED_IRR,
            "keeps_drawdown_not_above_baseline": max_drawdown <= dca_drawdown,
            "ready_for_more_aggressive_strategy": prediction_passed,
        },
    }
    with (REPORT_DIR / "module_scorecard.json").open("w") as f:
        json.dump(scorecard, f, indent=2)
    return scorecard


def write_reports(
    dates,
    prices,
    prediction_stats: Dict[str, object],
    leakage_stats: Dict[str, object],
    factor_validation: Dict[str, object],
    strategy: Dict[str, object],
    fixed_threshold: Dict[str, object],
    model_strategy: Dict[str, object],
    oracle: Dict[str, object],
    worst_case: Dict[str, object],
    sweep: Dict[str, object],
    dca: Dict[str, object],
) -> Dict[str, object]:
    """写出所有回测产物：权益曲线、场景分析、订单建议、摘要 JSON 和模块评分。"""

    equity_curve = []
    for strategy_row, dca_row in zip(strategy["curve"], dca["curve"]):
        row = dict(strategy_row)
        row["dca_equity"] = dca_row["dca_equity"]
        row["dca_drawdown"] = dca_row["dca_drawdown"]
        equity_curve.append(row)
    save_csv(REPORT_DIR / "equity_curve.csv", equity_curve, list(equity_curve[0].keys()))

    scenario_rows = []
    for oracle_row, worst_row in zip(oracle["curve"], worst_case["curve"]):
        scenario_rows.append({
            "date": oracle_row["date"],
            "oracle_equity": oracle_row["equity"],
            "oracle_drawdown": oracle_row["drawdown"],
            "worst_case_equity": worst_row["equity"],
            "worst_case_drawdown": worst_row["drawdown"],
        })
    save_csv(REPORT_DIR / "scenario_analysis.csv", scenario_rows, list(scenario_rows[0].keys()))
    save_csv(REPORT_DIR / "strategy_sweep.csv", sweep["rows"], list(sweep["rows"][0].keys()))

    latest_i = len(dates) - 1
    latest_equity = portfolio_value(strategy["holdings"], prices, latest_i)
    orders = build_orders(prices, latest_i, strategy["holdings"], latest_equity + MONTHLY_CONTRIBUTION, strategy["latest_weights"])
    with (REPORT_DIR / "latest_orders.json").open("w") as f:
        json.dump({
            "as_of": dates[-1].isoformat(),
            "next_monthly_contribution": MONTHLY_CONTRIBUTION,
            "orders": orders,
        }, f, indent=2)

    strategy_summary = strategy["summary"]
    dca_summary = dca["summary"]
    risk_controls = derive_risk_controls(strategy, oracle, worst_case, dca)
    module_scorecard = write_module_scorecard(prediction_stats, leakage_stats, factor_validation, strategy, fixed_threshold, dca, oracle, sweep)
    summary = {
        "as_of": dates[-1].isoformat(),
        "monthly_contribution": MONTHLY_CONTRIBUTION,
        "target_annualized_irr": TARGET_ANNUALIZED_IRR,
        "strategy": strategy_summary,
        "model_score_return_scenario": model_strategy["summary"],
        "fixed_threshold_scenario": fixed_threshold["summary"],
        "dca_benchmark": dca_summary,
        "oracle_scenario": oracle["summary"],
        "worst_case_scenario": worst_case["summary"],
        "strategy_sweep": {
            "best": sweep["best"],
            "feasible_count": sweep["feasible_count"],
            "total_count": sweep["total_count"],
            "note": "Sweep compares rule-signal, score-return allocation variants, and the fixed-threshold rule; it is diagnostics, not prediction training.",
        },
        "risk_controls": risk_controls,
        "factor_validation": factor_validation,
        "prediction_metrics": prediction_stats,
        "leakage_audit": leakage_stats,
        "module_scorecard": module_scorecard,
        "beats_dca_annualized_irr": strategy_summary["annualized_irr"] > dca_summary["annualized_irr"],
        "beats_dca_total_return": strategy_summary["total_return_on_principal"] > dca_summary["total_return_on_principal"],
        "max_drawdown_below_dca": strategy_summary["max_drawdown"] <= dca_summary["max_drawdown"],
        "meets_drawdown_target": strategy_summary["max_drawdown"] <= MAX_DRAWDOWN_TARGET,
        "meets_30pct_annualized_target": strategy_summary["annualized_irr"] >= TARGET_ANNUALIZED_IRR,
        "universe": UNIVERSE,
        "leakage_control": {
            "label": f"Each label uses a future {FORWARD_RETURN_DAYS}-calendar-day window mapped to actual trading days, and is used only for training later months.",
            "walk_forward": "For each date and asset, the model trains only on earlier observations for that asset.",
            "strategy": "Strategy consumes only current-month predicted scores and expected returns.",
        },
        "logic": {
            "module_1_prediction": "Prediction module supports pluggable estimators: rule_signal is the main estimator, model_walk_forward is retained as a model benchmark.",
            "module_2_strategy": "Main strategy uses legacy QQQ score thresholds; model score-return allocation is reported as a scenario, not the default strategy.",
            "module_3_backtest": "Reports compare strategy vs QQQ DCA plus oracle and worst-case scenarios.",
        },
    }
    with (REPORT_DIR / "backtest_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_backtest() -> Dict[str, object]:
    """完整回测入口：验证因子、加载行情、生成 walk-forward 预测、跑策略并输出报告。"""

    REPORT_DIR.mkdir(exist_ok=True)
    factor_validation = validate_factor_file()
    dates, prices = load_market_data()
    start_i = SLOW_TREND_WINDOW + 52

    observations = build_asset_observations(dates, prices, start_i)
    predictions = run_prediction_strategy(observations, "rule_signal")
    model_predictions = run_walk_forward_predictions(observations)
    prediction_stats = save_prediction_outputs(predictions)
    model_prediction_stats = save_prediction_outputs(model_predictions, "model")
    prediction_stats["model_benchmark"] = model_prediction_stats
    leakage_stats = leakage_audit(predictions)
    strategy = run_allocated_strategy(dates, prices, predictions, "legacy_signal_strategy")
    model_strategy = run_allocated_strategy(dates, prices, model_predictions, "score_return_strategy")
    fixed_threshold = run_allocated_strategy(dates, prices, predictions, "fixed_threshold_strategy")
    oracle = run_allocated_strategy(dates, prices, predictions, "oracle")
    worst_case = run_allocated_strategy(dates, prices, predictions, "worst_case")
    dca = run_monthly_qqq_dca(dates, prices, predictions[0].price_i)
    sweep = strategy_sweep(dates, prices, predictions, dca["summary"]["max_drawdown"])
    return write_reports(dates, prices, prediction_stats, leakage_stats, factor_validation, strategy, fixed_threshold, model_strategy, oracle, worst_case, sweep, dca)
