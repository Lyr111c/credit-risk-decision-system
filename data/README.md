# 数据说明（中文）

## 数据集概览

- 名称：Default of Credit Card Clients（信用卡客户次月违约数据集）
- 创建者：I-Cheng Yeh
- 提供方：UCI Machine Learning Repository
- 数据集页面：<https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- 数据集 DOI：<https://doi.org/10.24432/C55S3H>
- 原论文 DOI：<https://doi.org/10.1016/j.eswa.2007.12.020>
- 许可证：[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)
- 样本范围：30,000 名中国台湾地区信用卡客户，包含 23 个解释变量、1 个客户编号和 1 个二元预测目标
- 预测目标：客户是否在下一个月发生违约（`1` = 是，`0` = 否）

数据来源和许可证于 2026-07-17 完成核验。本仓库不直接分发原始数据。将原始文件排除在 Git 之外，可以明确数据来源，并避免把数据文件的历史版本写入代码仓库。

## 下载原始数据

安装项目依赖后，在仓库根目录运行：

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data download
```

该命令从 UCI 官方地址下载 ZIP 文件，验证 SHA-256，并把原始 Excel 文件解压到 `data/raw/`。两个文件均由 `.gitignore` 排除，不会被提交到 GitHub。

已验证的源文件：

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
| `default_of_credit_card_clients.zip` | 5,539,494 | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| `default of credit card clients.xls` | 5,539,328 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |

这些哈希值对应 2026-07-17 从 UCI 官方地址取得的文件。如果 UCI 有意替换文件，不应绕过校验；应先检查新文件，并通过代码审查更新数据来源记录。

## 本地目录结构

```text
data/
|-- raw/          # 未经修改的下载文件
|-- interim/      # 可选：可复现的中间数据
`-- processed/    # 可选：可直接用于建模的数据
```

Git 只跟踪 `data/README.md`。除非后续评审明确允许重新分发，否则原始数据、中间数据和处理后表格只保留在本地。

## 先理解“信用风险决策”

这里的“违约”是数据集对下个月未按约履行付款义务的结果标记。模型的任务是综合客户属性和过去六个月的账户行为，估计下个月违约的可能性。这个概率可以支持风险排序、人工复核或额度管理，但不能单独代替完整的审批政策。

变量与违约之间的统计关联不等于因果关系。单个变量通常也不足以决定批准或拒绝。例如，账单金额高可能只是消费较多；如果客户同时按时、足额还款，风险含义会与连续逾期且还款很少的情况完全不同。

## 变量通俗解释

### 客户编号与预测目标

| 原始变量 | 通俗含义 | 如何影响决策或模型 | 重要提醒 |
|---|---|---|---|
| `ID` | 每行客户记录的唯一编号，类似表格中的行号 | 只用于定位、连接和检查记录，不应作为预测依据 | 编号大小没有金融含义，必须从模型特征中排除 |
| `default payment next month` | 客户下个月是否违约：`1` 表示违约，`0` 表示未违约 | 这是模型要预测的结果，不是做预测时可以提前知道的输入 | 数据中违约率为 22.12%；把它放入输入会造成数据泄漏 |

### 客户与账户基本信息

| 原始变量 | 通俗含义与取值 | 如何影响决策或模型 | 重要提醒 |
|---|---|---|---|
| `LIMIT_BAL` | 银行已经授予客户的信用额度，单位为新台币；可理解为客户最多可以使用的信用规模 | 应结合账单余额判断额度使用压力。例如，同样是 50,000 元账单，对 60,000 元额度和 500,000 元额度代表的压力不同 | 这是既有客户已经获得的额度，包含历史审批结果；额度高不等于客户未来一定低风险 |
| `SEX` | 数据源记录的性别：`1` = 男，`2` = 女 | 可能与历史样本中的违约率相关，但不应被解释为性别导致违约 | 属于敏感人口属性。使用前应评估合法性、公平性和不同群体的模型表现，不能据此直接拒绝客户 |
| `EDUCATION` | 教育程度：`1` = 研究生，`2` = 大学，`3` = 高中，`4` = 其他；数据还出现未说明的 `0`、`5`、`6` | 可能提供有限的群体差异信息，但通常不是直接、稳定的还款能力指标 | `0`、`5`、`6` 的含义不明，不能擅自命名；教育也是需要谨慎使用的人口属性 |
| `MARRIAGE` | 婚姻状态：`1` = 已婚，`2` = 未婚，`3` = 其他；数据还出现未说明的 `0` | 模型可能观察到不同群体的历史差异，但婚姻状态本身不能证明偿债能力 | `0` 的含义不明；该变量可能带来公平性和隐私风险，不应形成自动拒绝规则 |
| `AGE` | 客户年龄，单位为岁，样本范围为 21 至 79 岁 | 年龄可能与收入阶段、信用历史长度等因素共同关联，但必须与实际账户行为一起判断 | 年龄属于敏感属性。不能把相关性解释为年龄导致风险，也要检查不同年龄群体的误判率 |

### 过去六个月的还款状态

还款状态表示客户是否按时还款以及逾期了多久。UCI 只明确说明 `-1` 表示按时还款，正整数表示逾期月数；正数越大，表示拖欠时间越长。原始文件还包含 `-2` 和 `0`，但 UCI 页面没有定义它们，因此本项目保留这两个类别，不自行猜测含义。

| 原始变量 | 对应月份 | 通俗含义 | 如何影响决策或模型 |
|---|---|---|---|
| `PAY_0` | 2005 年 9 月 | 最近一个月的还款状态 | 最近且严重的逾期通常是较直接的风险信号；按时还款通常是较积极的信号 |
| `PAY_2` | 2005 年 8 月 | 前第二个月的还款状态 | 与 9 月状态结合，可判断问题是偶发还是连续发生 |
| `PAY_3` | 2005 年 7 月 | 前第三个月的还款状态 | 连续多个正值通常比单月异常更值得关注 |
| `PAY_4` | 2005 年 6 月 | 前第四个月的还款状态 | 帮助模型识别中期还款习惯和逾期持续时间 |
| `PAY_5` | 2005 年 5 月 | 前第五个月的还款状态 | 与较近月份比较，可观察风险是否改善或恶化 |
| `PAY_6` | 2005 年 4 月 | 前第六个月的还款状态 | 提供更早的行为背景，但一般不能脱离近期信息单独解释 |

不能把 `PAY_*` 简单当作连续金额。它们是有顺序含义的状态编码，而且 `-2`、`0` 的含义未知。建模和报告时应明确编码处理方式。

### 过去六个月的账单金额

账单金额是每个月账单周期结束时，账户显示的应付或结余金额，单位为新台币。它不是客户的收入，也不是本月实际还款额。负数可能表示溢缴、退款或账务调整，不能在没有证据时当作错误删除。

| 原始变量 | 对应月份 | 通俗含义 | 如何影响决策或模型 |
|---|---|---|---|
| `BILL_AMT1` | 2005 年 9 月 | 最近一期账单金额 | 应与 `LIMIT_BAL`、`PAY_AMT1` 和还款状态共同判断；接近额度且还款不足可能表示较大压力 |
| `BILL_AMT2` | 2005 年 8 月 | 前第二期账单金额 | 与最近账单比较，可以观察余额是上升、下降还是稳定 |
| `BILL_AMT3` | 2005 年 7 月 | 前第三期账单金额 | 帮助识别持续累积的余额，而不是只看一个月的偶然高消费 |
| `BILL_AMT4` | 2005 年 6 月 | 前第四期账单金额 | 为中期余额趋势提供信息 |
| `BILL_AMT5` | 2005 年 5 月 | 前第五期账单金额 | 可与付款和额度结合，观察客户是否在逐步降低债务 |
| `BILL_AMT6` | 2005 年 4 月 | 前第六期账单金额 | 提供六个月观察窗口的起点，便于比较整体变化 |

账单金额高本身不等于高风险。更有意义的组合包括账单占额度的比例、账单变化趋势，以及账单之后是否有相应还款。

### 过去六个月的实际还款金额

实际还款金额表示客户在对应月份支付了多少，单位为新台币。金额为 `0` 表示该月记录中没有付款，但是否构成违约仍需结合账单和还款状态判断。

| 原始变量 | 对应月份 | 通俗含义 | 如何影响决策或模型 |
|---|---|---|---|
| `PAY_AMT1` | 2005 年 9 月 | 最近一个月的实际还款额 | 相对账单较充足且持续的付款通常是积极信号；付款很少需结合是否存在应还账单判断 |
| `PAY_AMT2` | 2005 年 8 月 | 前第二个月的实际还款额 | 与账单和 9 月付款结合，可判断还款是否持续 |
| `PAY_AMT3` | 2005 年 7 月 | 前第三个月的实际还款额 | 帮助识别偶发大额付款与稳定付款习惯的差别 |
| `PAY_AMT4` | 2005 年 6 月 | 前第四个月的实际还款额 | 为中期偿付行为提供信息 |
| `PAY_AMT5` | 2005 年 5 月 | 前第五个月的实际还款额 | 可用于观察付款能力或意愿是否随时间变化，但不能单独证明原因 |
| `PAY_AMT6` | 2005 年 4 月 | 前第六个月的实际还款额 | 提供较早的付款行为背景，用于形成完整的六个月历史 |

还款金额越大不一定越好，因为它必须相对于账单金额理解。例如，没有账单时付款为 `0` 可能很正常；账单很高但长期只支付很少，则可能反映更高压力。

## 一个简化的综合判断示例

假设两位客户最近一期账单都是 50,000 新台币：

- 客户 A 的额度为 500,000，过去六个月均按时还款，且最近付款覆盖了大部分账单。
- 客户 B 的额度为 60,000，最近连续逾期，且每月付款相对账单很少。

“账单金额 50,000”这个单一变量对两人的含义不同。模型应综合额度、账单、付款和逾期轨迹估计风险，同时检查人口属性是否造成不公平影响。最终决策还需要结合阈值、人工复核、法规和机构政策；模型分数不是对个人品质的评价，也不应成为无法解释的自动拒绝理由。

## 数据源特有注意事项

- 工作簿有两行表头，规范字段名位于第二行，因此必须使用 `header=1` 读取。
- UCI 说明数据没有缺失值；正确解析后，下载文件确实没有空单元格。
- 文件包含未说明的类别编码：`EDUCATION` 中有 `0`、`5`、`6`，`MARRIAGE` 中有 `0`，还款状态中有 `-2` 和 `0`。本项目保留并标记它们，不会静默重编码。
- 去掉 `ID` 后有 35 行记录完全相同，但所有 ID 都唯一。不同客户可能拥有相同的观测值和目标，因此不自动删除这些记录。
- 账单金额存在负数，可能代表账户余额为贷方、溢缴、退款或调整，因此不自动视为无效。

## 业务使用边界

- 数据描述的是 2005 年中国台湾地区的既有信用卡客户，不代表当前中国大陆银行或消费金融的新申请客户。
- 历史账单和还款行为通常无法在新客户首次申请时取得。它们适合本项目的既有客户行为风险预测，但限制了对真正贷前审批的结论。
- 数据不支持可靠的跨时期开发与验证设计。随机切分不能称为样本外时间验证（out-of-time validation）。
- 无法从这些数据恢复拒绝推断、宏观经济周期、资金成本、干预后的额度变化或监管决策规则。
- 本项目中的审批、收入、损失或阈值分析只能是透明的教学假设，不能声称为实测业务结果。

字段的项目命名、观测范围和建模处理说明见 [`references/data_dictionary.md`](../references/data_dictionary.md)。

---

# Data Documentation (English)

## Dataset overview

- Name: Default of Credit Card Clients
- Creator: I-Cheng Yeh
- Provider: UCI Machine Learning Repository
- Dataset page: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- Dataset DOI: <https://doi.org/10.24432/C55S3H>
- Research paper DOI: <https://doi.org/10.1016/j.eswa.2007.12.020>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)
- Scope: 30,000 Taiwan credit-card client records, 23 explanatory variables, one client identifier, and one binary target
- Target: default payment in the following month (`1` = yes, `0` = no)

The source page and license were checked on 2026-07-17. The repository does not redistribute the raw dataset. Keeping source files out of Git makes provenance explicit and prevents data-file history from accumulating in the code repository.

## Download the raw data

After installing the project dependencies, run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data download
```

The command downloads the official UCI ZIP, verifies its SHA-256 digest, and extracts the original Excel file into `data/raw/`. Both files are excluded by `.gitignore` and will not be committed to GitHub.

Verified source artifacts:

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| `default_of_credit_card_clients.zip` | 5,539,494 | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| `default of credit card clients.xls` | 5,539,328 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |

The hashes describe files retrieved from the official UCI URL on 2026-07-17. If UCI intentionally replaces an artifact, do not bypass the checksum. Inspect the new file and update the recorded provenance in a reviewed commit.

## Local layout

```text
data/
|-- raw/          # Unmodified downloaded artifacts
|-- interim/      # Optional reproducible intermediate outputs
`-- processed/    # Optional model-ready outputs
```

Only `data/README.md` is versioned. Raw, intermediate, and processed tabular files remain local unless a later review explicitly approves redistribution.

## What a credit-risk decision means here

In this dataset, default is the recorded outcome that a client did not meet the following month's payment obligation as defined by the source. A model combines customer attributes with six months of account behavior to estimate the probability of default in the next month. That probability can support risk ranking, manual review, or credit-limit management, but it does not replace a complete decision policy.

A statistical association with default is not proof of causation. One variable is also rarely enough to approve or decline someone. A large bill, for example, may simply reflect high spending. Its risk meaning is very different when the client pays on time and in substantial amounts than when the client is repeatedly late and pays very little.

## Plain-language variable guide

### Client identifier and prediction target

| Raw variable | Plain-language meaning | Possible role in a decision or model | Important caution |
|---|---|---|---|
| `ID` | A unique number for each client record, similar to a row number | Used only to locate, join, and audit records; it should not influence a prediction | The number has no financial meaning and must be excluded from model features |
| `default payment next month` | Whether the client defaults in the next month: `1` means default and `0` means no default | This is the outcome the model predicts, not information available as an input at decision time | The observed default rate is 22.12%; using the target as an input would be data leakage |

### Client and account information

| Raw variable | Plain-language meaning and values | Possible role in a decision or model | Important caution |
|---|---|---|---|
| `LIMIT_BAL` | The credit limit already granted by the bank, in New Taiwan dollars; it is the scale of credit available to the client | Interpret it with bill balances to understand pressure on available credit. A 50,000 bill means something different against a 60,000 limit than against a 500,000 limit | This existing-client limit contains information from earlier underwriting. A high limit does not guarantee low future risk |
| `SEX` | Gender as recorded by the source: `1` = male and `2` = female | It may be associated with default rates in the historical sample, but that does not mean gender causes default | This is a sensitive demographic attribute. Check legality, fairness, and group performance; never use it as a direct automatic decline rule |
| `EDUCATION` | Education level: `1` = graduate school, `2` = university, `3` = high school, `4` = others; undocumented `0`, `5`, and `6` also occur | It may contain limited group-level information but is not a direct or stable measure of repayment ability | Do not invent meanings for `0`, `5`, or `6`. Education is also a demographic attribute that requires cautious use |
| `MARRIAGE` | Marital status: `1` = married, `2` = single, `3` = others; undocumented `0` also occurs | A model may observe historical group differences, but marital status does not prove ability to repay | The meaning of `0` is unknown. The field can introduce fairness and privacy concerns and should not define an automatic decline rule |
| `AGE` | Client age in years; observed from 21 to 79 | Age may be associated with life stage, income stage, or length of credit history, but it should be interpreted with actual account behavior | Age is sensitive. Do not treat correlation as causation, and compare error rates across age groups |

### Repayment status over the previous six months

Repayment status records whether the client paid on time and, when late, how long payment was delayed. UCI explicitly defines `-1` as paid duly and positive integers as months of delay; larger positive numbers indicate longer delays. The workbook also contains `-2` and `0`, but the UCI page does not define them. This project preserves those categories without guessing their meaning.

| Raw variable | Month | Plain-language meaning | Possible role in a decision or model |
|---|---|---|---|
| `PAY_0` | September 2005 | Repayment status for the most recent month | Recent and severe delay is usually a direct adverse signal; timely payment is usually a positive signal |
| `PAY_2` | August 2005 | Repayment status two months back | Combined with September, it helps distinguish a one-off issue from repeated delay |
| `PAY_3` | July 2005 | Repayment status three months back | Positive values across several months are generally more concerning than one isolated event |
| `PAY_4` | June 2005 | Repayment status four months back | Helps describe medium-term repayment habits and the persistence of delay |
| `PAY_5` | May 2005 | Repayment status five months back | Comparison with recent months can indicate whether behavior is improving or deteriorating |
| `PAY_6` | April 2005 | Repayment status six months back | Adds earlier behavioral context but should not be interpreted without recent information |

Do not treat `PAY_*` as ordinary continuous monetary values. They are ordered status codes, and the meanings of `-2` and `0` are unknown. Modeling and reporting must state how these codes are handled.

### Bill statement amounts over the previous six months

A bill amount is the amount due or balance shown when a monthly statement closes, in New Taiwan dollars. It is not the client's income and it is not the amount actually paid that month. Negative values may represent credits, overpayments, refunds, or accounting adjustments and should not be removed as errors without evidence.

| Raw variable | Month | Plain-language meaning | Possible role in a decision or model |
|---|---|---|---|
| `BILL_AMT1` | September 2005 | Most recent statement amount | Interpret with `LIMIT_BAL`, `PAY_AMT1`, and repayment status; a balance near the limit with insufficient payment may indicate greater pressure |
| `BILL_AMT2` | August 2005 | Statement amount two months back | Comparison with the latest bill shows whether balances are rising, falling, or stable |
| `BILL_AMT3` | July 2005 | Statement amount three months back | Helps identify balances that accumulate over time rather than a single month of high spending |
| `BILL_AMT4` | June 2005 | Statement amount four months back | Adds information about the medium-term balance trend |
| `BILL_AMT5` | May 2005 | Statement amount five months back | Combined with payments and limit, it may show whether the client is reducing debt |
| `BILL_AMT6` | April 2005 | Statement amount six months back | Provides the start of the six-month window for an overall trend comparison |

A high bill alone does not imply high risk. More meaningful combinations include the bill relative to the credit limit, the direction of balances over time, and whether corresponding payments were made.

### Actual payment amounts over the previous six months

Actual payment amount is how much the client paid in the corresponding month, in New Taiwan dollars. A value of `0` means that no payment is recorded for that month, but whether this constitutes a problem depends on the bill and repayment status.

| Raw variable | Month | Plain-language meaning | Possible role in a decision or model |
|---|---|---|---|
| `PAY_AMT1` | September 2005 | Actual payment in the most recent month | Payments that are substantial relative to bills and sustained over time are generally positive; a small payment matters only when an amount was due |
| `PAY_AMT2` | August 2005 | Actual payment two months back | Combined with the bill and September payment, it shows whether repayment is sustained |
| `PAY_AMT3` | July 2005 | Actual payment three months back | Helps distinguish a one-off large payment from a stable repayment pattern |
| `PAY_AMT4` | June 2005 | Actual payment four months back | Adds information about medium-term payment behavior |
| `PAY_AMT5` | May 2005 | Actual payment five months back | May show changing ability or willingness to pay, but cannot establish the reason by itself |
| `PAY_AMT6` | April 2005 | Actual payment six months back | Provides earlier payment context for the complete six-month history |

A larger payment is not automatically better because it must be interpreted relative to the bill. A zero payment may be normal when no amount is due; persistently small payments against large bills may indicate greater pressure.

## A simplified combined-decision example

Suppose two clients both have a latest bill of NT$50,000:

- Client A has a NT$500,000 limit, paid on time throughout the previous six months, and covered most of the latest bill.
- Client B has a NT$60,000 limit, has recently been late for several consecutive months, and pays little relative to each bill.

The single value “bill amount = 50,000” has a different meaning for these clients. A model should combine limits, bills, payments, and delay trajectories, while checking whether demographic attributes create unfair effects. The final decision must also reflect thresholds, manual review, law, and institutional policy. A model score is not a judgment of personal character and should not become an unexplained automatic decline.

## Source-specific caveats

- The workbook has two header rows. Canonical column names are on the second row, so it must be read with `header=1`.
- UCI documents no missing values, and the downloaded file contains no null cells after correct parsing.
- The file contains undocumented category codes: `EDUCATION` includes `0`, `5`, and `6`; `MARRIAGE` includes `0`; repayment status includes `-2` and `0`. They are retained and flagged instead of silently recoded.
- There are 35 identical rows after dropping `ID`, but all IDs are unique. Different clients can share the same observed values and target, so these records are not automatically deleted.
- Negative bill amounts may represent credit balances, overpayments, refunds, or adjustments. They are not automatically treated as invalid.

## Business-use limitations

- The records describe existing Taiwan credit-card clients in 2005, not current mainland Chinese banking or consumer-finance applicants.
- Historical billing and repayment variables may not be available in a new-customer application workflow. They are suitable for this existing-customer behavioral-risk exercise but limit claims about true application underwriting.
- The dataset does not provide a reliable multi-period development and validation design. A random split must not be called out-of-time validation.
- Reject inference, macroeconomic cycles, funding costs, credit limits after intervention, and regulatory decision rules cannot be recovered from these data.
- Any approval, revenue, loss, or threshold analysis in this project is a transparent teaching scenario, not a measured business result.

See [`references/data_dictionary.md`](../references/data_dictionary.md) for project field names, observed ranges, and modeling-treatment notes.
