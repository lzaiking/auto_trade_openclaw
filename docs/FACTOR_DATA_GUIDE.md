# Factor Data Guide

## 目标

`data/factors/monthly_factors.csv` 是预测模块的统一月度数据表。每一行代表一个月度决策日，包含当时可见的公开因子、三个 ETF 的价格特征，以及只能用于事后评估的未来 label。

生成命令：

```bash
python3 code/build_monthly_factors.py
```

## 当前保留字段

| 字段 | 范围 | 含义 |
| --- | ---: | --- |
| `date` | `YYYY-MM-DD` | 当月第一个交易日 |
| `breadth_200dma` | `0` 到 `1` | `QQQ / GLD / SGOV` 三个可交易 ETF 中，高于自身 200 日均线的比例 |
| `tbill_3m_rate` | 小数 | FRED `DTB3` 3 个月 T-bill 日度利率，按决策日或之前最近可用值填充 |

## 价格数据列

脚本会把本地价格缓存 `data/prices/*.csv` 整合进同一张月度表：

- `{asset}_close`
- `{asset}_extension_200dma`
- `{asset}_rsi_14`
- `{asset}_return_63d`

其中 `asset` 包括：

- `qqq`
- `gld`
- `sgov`

## 后验 Label 列

每个资产会生成四个后验 label：

- `label_{asset}_forward_30d_return`
- `label_{asset}_forward_30d_avg_return`
- `label_{asset}_forward_30d_up`
- `label_{asset}_forward_30d_avg_up`

label 使用未来 30 个自然日构造：如果第 30 天不是交易日，就取之后第一个可用交易日作为端点；均价 label 使用这段窗口内所有实际交易日价格均值。最近月份如果还没有足够未来数据，label 必须留空。

## 已删除字段

以下字段暂时不进入 `monthly_factors.csv`：

- `pe`
- `pb`
- `eps`
- `fcf_yield_minus_tbill`
- `eps_revision_score`
- `forward_peg_score`
- `ai_profit_conversion_score`
- `top_weight_concentration`
- `crowding_score`

删除原因：这些字段如果要用于回测，需要历史月度、点位时序、可复现的数据。当前公开网页通常只提供最新估值或需要付费/API-key 的历史数据，直接硬填会引入未来函数或不可复现口径。后续如果接入可靠的历史点位数据源，可以按同一接口重新加入。

## 点位时序规则

- 预测特征只能使用决策日前已经可见的数据。
- 价格特征只能使用决策日当天及之前的价格。
- `label_*` 列只能用于预测评估，策略模块不能读取。
- 如果宏观数据当天缺失，使用决策日前最近一个可用值。
- 缺失值可以留空；验证报告会记录覆盖率。

## 最低可用标准

当前验证模块使用这些门槛：

- 必须包含所有 required columns。
- `date` 必须可解析且不能有重复月份。
- 数值必须落在字段范围内。
- 每个预测因子覆盖率建议达到 `80%` 以上。

覆盖率不足时，回测仍可运行，但 `module_scorecard.json` 会提示预测模块不够成熟。
