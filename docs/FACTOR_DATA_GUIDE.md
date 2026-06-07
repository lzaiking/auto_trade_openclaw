# Factor Data Guide

## 目标

`data/factors/monthly_factors.csv` 是预测模块的月度基本面/拥挤度因子输入。每一行代表一个月度决策日前已经可见的信息，不能用之后才发布或之后才修订的数据。

回测会自动生成：

- `report/factor_validation.json`
- `report/prediction_metrics.json`
- `report/module_scorecard.json`

只有当因子结构通过、覆盖率足够、预测 AUC 提升且防穿越审计通过时，策略模块才应该考虑更激进的仓位。

## 字段

| 字段 | 建议范围 | 含义 |
| --- | ---: | --- |
| `date` | `YYYY-MM-DD` | 当月决策日期，建议使用当月第一个交易日 |
| `eps_revision_score` | `-5` 到 `5` | EPS 预期上修强度，越高越利多 |
| `forward_peg_score` | `-5` 到 `5` | forward PEG 吸引力，越高越利多 |
| `fcf_yield_minus_tbill` | `-1` 到 `1` | FCF yield 减短债收益率，使用小数 |
| `ai_profit_conversion_score` | `-5` 到 `5` | AI capex 转化为利润的评分，越高越利多 |
| `top_weight_concentration` | `0` 到 `1` | 指数头部权重集中度 |
| `breadth_200dma` | `0` 到 `1` | 成分股高于 200 日均线的比例 |
| `crowding_score` | `-5` 到 `5` | 资金拥挤度/仓位热度评分 |

## 点位时序规则

- 只能填入当月决策日前已经可见的数据。
- 不要用之后修订过的历史 EPS、PEG、成分股权重或资金流数据覆盖旧行。
- 如果一个数据源发布有滞后，以实际发布时间为准，不以数据所属月份为准。
- 缺失值可以留空；模型会用训练集均值填充，但验证报告会记录覆盖率。

## 最低可用标准

当前验证模块使用这些门槛：

- 必须包含所有 required columns。
- `date` 必须可解析且不能有重复月份。
- 数值必须落在字段范围内。
- 每个因子覆盖率建议达到 `80%` 以上。

覆盖率不足时，回测仍可运行，但 `module_scorecard.json` 会把预测模块标记为未就绪。
