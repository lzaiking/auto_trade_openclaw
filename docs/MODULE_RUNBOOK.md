# Module Runbook

## 目录职责

- `code/market_data.py`：下载、缓存并对齐 `QQQ / GLD / SGOV` 日线价格，计算均线、RSI、涨幅等基础指标。
- `code/factor_validation.py`：检查 `data/factors/monthly_factors.csv` 的字段、取值范围和覆盖率。
- `code/prediction_module.py`：预测模块。先构造月度样本，再通过可插拔预估策略输出 `score`、`predicted_1m_return` 和后验评估。
- `code/strategy_module.py`：交易策略模块。只读取预测模块结果，按策略规则生成目标仓位并回测。
- `code/backtest_module.py`：回测报告模块。串联因子校验、预测、策略、基线、场景分析和报告输出。
- `code/trading_system.py`：主入口。

## 预测模块

预测模块使用策略设计模式。当前有两种预估策略：

- `rule_signal`：规则预估策略，直接用 QQQ/GLD/SGOV 的 200 日均线偏离、14 日 RSI、63 日涨幅生成分数。当前主交易策略默认使用这套预估。
- `model_walk_forward`：线性模型预估策略，保留 walk-forward 训练逻辑，用作模型对照和后续扩展。

主流程会生成：

- `report/factor_predictions.csv`：规则预估明细
- `report/prediction_metrics.json`：规则预估准确度
- `report/model_factor_predictions.csv`：模型预估明细
- `report/model_prediction_metrics.json`：模型预估准确度

## 策略模块

当前主交易策略是 `legacy_signal_strategy`：

- 默认以 QQQ 为核心资产
- 只消费预测模块输出的 `score`
- QQQ score 高时主要买 QQQ；QQQ score 低时按固定档位转入防守仓位
- 防守仓位优先给 GLD
- 如果 GLD score 低于 `0.50` 或 GLD 自身偏高，则防守仓位转入 SGOV

保留的对照策略：

- `rule_signal_strategy`：直接读取价格特征的阶梯式防守策略，作为旧实现对照
- `score_return_strategy`：使用模型分数和预测收益做多资产吸引力分配
- `fixed_threshold_strategy`：按 QQQ 预测收益分档配置防守仓位
- `oracle`：事后最优上限场景
- `worst_case`：事后最差场景
- `monthly_qqq_dca`：每月固定买入 QQQ 基线

## 运行命令

完整运行：

```bash
python3 code/trading_system.py
```

语法检查：

```bash
python3 -m py_compile code/trading_system.py code/market_data.py code/prediction_module.py code/strategy_module.py code/backtest_module.py code/factor_validation.py
```

检查格式空白：

```bash
git diff --check
```

检查所有函数是否都有说明：

```bash
python3 -c 'import ast, pathlib, sys; missing=[]; [missing.extend([f"{path}:{node.name}" for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.FunctionDef) and ast.get_docstring(node) is None]) for path in sorted(pathlib.Path("code").glob("*.py"))]; print("missing docstrings:", missing); sys.exit(1 if missing else 0)'
```

## 主要输出

- `report/backtest_summary.json`：完整回测汇总
- `report/equity_curve.csv`：主策略和 QQQ 定投权益曲线
- `report/latest_orders.json`：下一次月供后的调仓建议
- `report/strategy_sweep.csv`：策略参数和规则对照
- `report/scenario_analysis.csv`：oracle / worst-case 场景
- `report/module_scorecard.json`：预测、策略、回测三个模块评分

## 扩展方式

新增预测方法时，在 `prediction_module.py` 里新增一个继承 `PredictionEstimator` 的类，并在 `run_prediction_strategy` 的 registry 里注册。

新增交易策略时，在 `strategy_module.py` 里新增一个继承 `AllocationStrategy` 的类，并在 `target_weights_from_predictions` 的 registry 里注册。
