# Trading System Prototype

这是一个可跑的原型，目标是把“每月固定投入 10,000 到 QQQ”升级成预测驱动的月度配置策略，并把系统拆成三个可独立评估的模块：

1. 预测模块：用月度因子预测未来 30 个自然日窗口收益，并评估准确度
2. 策略模块：只读取 `QQQ / GLD / SGOV` 的预测分数和预期收益，按固定规则决定仓位，不参与模型训练
3. 回测模块：评估收益、回撤、AUC、oracle 上限和 worst-case 风险

## 文件

- `code/trading_system.py`：主脚本，负责数据读取、walk-forward 预测、策略回测、报告输出
- `code/factor_validation.py`：月度因子结构、范围和覆盖率验证
- `code/prediction_module.py`：预测模块
- `code/strategy_module.py`：策略模块
- `code/backtest_module.py`：回测报告模块
- `code/market_data.py`：行情、缓存、指标和共享工具
- `data/factors/monthly_factors.csv`：月度因子输入模板
- `report/backtest_summary.json`：回测汇总（含策略与每月定投 QQQ 基准对比）
- `report/factor_validation.json`：月度因子结构、取值范围、覆盖率和点位时序提醒
- `report/prediction_metrics.json`：预测模型准确度，包含 30 自然日端点价 AUC 和窗口内交易日均价 AUC
- `report/prediction_calibration.csv`：分数分桶校准，检查高分月份是否优于低分月份
- `report/leakage_audit.json`：防穿越审计
- `report/module_scorecard.json`：预测、策略、回测三个模块的独立评分
- `report/factor_predictions.csv`：每月预测、后验 label、误差和模型系数
- `report/equity_curve.csv`：权益曲线（含策略、月度定投 QQQ、预测值和仓位）
- `report/scenario_analysis.csv`：oracle 和 worst-case 场景曲线
- `report/strategy_sweep.csv`：score+return 策略和固定阈值对照的诊断表
- `report/latest_orders.json`：下一次月度投入后的调仓差额建议

## 策略逻辑

### 1) 预测模块
- 输入因子：EPS 修正、forward PEG、FCF yield 相对短债、AI capex 利润兑现、指数集中度、市场宽度、拥挤度，以及资产自己的价格类兜底因子。
- 后验 label：当前日期往后 30 个自然日，若当天休市则取之后第一个可用交易日作为端点；均价口径使用这段自然日窗口内所有实际交易日价格均值。
- 输出：`predicted_1m_return`、0-1 上涨分数、端点价后验收益、均价后验收益、误差和模型系数；0-1 分数会在方向概率、收益幅度分数和混合分数之间，按历史 OOS 表现选择。
- 评估：整体端点价 AUC、30 自然日窗口均价 AUC、每资产 AUC、每年 AUC、MAE/MSE、分数分桶校准。

### 2) 策略模块
- 标的池：`QQQ, GLD, SGOV`
- 核心资产：`QQQ`
- 替代资产：`GLD`
- 防守资产：`SGOV`
- 每月第一个交易日投入 10,000，并按三个资产的预测分数和预期收益再平衡。
- 当前默认至少保留 85% QQQ，GLD 上限 15%，SGOV 上限 20%。
- 若 GLD 偏离 200 日均线达到 10% 以上或 RSI 达到 70 以上，则 GLD 吸引力按 0 处理。
- 当前预测 AUC 较弱，所以策略保守使用分数和收益信号，避免把弱预测放大。
- 同时输出 oracle 和 worst-case 场景，用来评估理论上限和极端错误风险。

## 因子数据

因子填写规则见 `docs/FACTOR_DATA_GUIDE.md`。当前 `monthly_factors.csv` 只有表头，所以报告会显示基本面因子覆盖率为 0%；这也是现阶段策略无法跑赢 QQQ 定投的主要瓶颈。

### 3) 自动交易系统
当前版本默认输出“建议订单”，不直接连真实券商下单，以免误触发真实交易。
如果要实盘接券商，可以在下一版对接：
- Alpaca
- Interactive Brokers
- Tiger / 富途（若有可用 API）

## 运行

```bash
python3 code/trading_system.py
```

## 备注

- 数据源优先使用本地缓存 `data/prices/`，缓存过期后先尝试 Yahoo Finance chart API，再用 Stooq 兜底
- Yahoo 请求遇到 429 限流时会退避重试；如果线上数据源失败但本地有旧缓存，会使用旧缓存继续回测
- 这是一个研究/原型版本，不等于可直接实盘的最终系统
- 下一步建议：加入交易成本、滑点、Walk-forward 验证、参数稳健性测试
