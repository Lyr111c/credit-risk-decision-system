# Credit Risk Decision System

## 中文

基于公开信用风险数据，构建从数据审计、违约概率建模、评分卡、模型比较与概率校准，到审批阈值和假设风险收益模拟的可复现流程。

> **状态：** 阶段一至四已完成，包括数据来源核验、数据质量审计、探索性数据分析、总体违约率虚拟基线和逻辑回归基线。WOE/IV 评分卡、Challenger 模型、策略模拟和监控设计尚未实施。

### 业务问题

在假设误批违约客户会产生信用损失、误拒优质客户会损失潜在收益的条件下，如何选择审批阈值，在控制坏账风险的同时维持合理通过率？本项目中的收益、损失和审批策略均为教学用途的透明假设，不代表真实金融机构的经营参数或授信决策。

### 数据来源与边界

第一版使用 UCI Machine Learning Repository 的 [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) 数据集（CC BY 4.0），包含 30,000 条记录，实际违约率为 22.12%。原始数据不提交到仓库；下载与校验方式记录在 [`data/README.md`](data/README.md)，字段定义见 [`references/data_dictionary.md`](references/data_dictionary.md)。

- 数据来自公开教学与研究场景，不代表当前银行或消费金融业务；
- 历史账单、付款和还款状态适用于已有客户行为风险预测，不等同于新客户贷前审批；
- 数据不支持严格跨时期验证，分层随机测试集不是 OOT；
- 相关性和组间违约率差异不解释为因果影响；
- `customer_id` 仅为标识，不进入模型。

### 已完成方法

使用固定随机种子 `42` 将数据按目标变量分层切分为训练集 60%、验证集 20% 和测试集 20%。类别变量通过 `OneHotEncoder(handle_unknown="ignore")` 编码，数值变量通过 `StandardScaler` 标准化；全部预处理只在训练集拟合。比较以下三个模型：

1. `DummyClassifier(strategy="prior")` 总体违约率虚拟基线；
2. 无权重 Logistic Regression，作为主要可解释基线；
3. `class_weight="balanced"` Logistic Regression，作为不平衡敏感性分析。

模型使用 ROC-AUC、PR-AUC、KS、Brier Score、校准曲线，以及固定 `0.5` 阈值下的混淆矩阵、Precision 和 Recall 评估。Accuracy 不是主要指标，`0.5` 也不是业务审批阈值。

### 实际测试集结果

| 模型 | ROC-AUC | PR-AUC | KS | Brier Score | Precision @ 0.5 | Recall @ 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 总体违约率虚拟基线 | 0.5000 | 0.2212 | 0.0000 | 0.1723 | 0.0000 | 0.0000 |
| 无权重逻辑回归 | 0.7591 | 0.5258 | 0.3970 | 0.1388 | 0.6662 | 0.3519 |
| 类别平衡逻辑回归 | 0.7590 | 0.5226 | 0.3946 | 0.1857 | 0.4848 | 0.5531 |

无权重模型的概率质量优于类别平衡版本；加权版本在 `0.5` 阈值下提高 Recall，但同时降低 Precision 并产生更多假阳性。这是统计取舍，不代表业务最优模型。完整结果见 [`reports/model_report.md`](reports/model_report.md)，EDA 见 [`reports/eda_report.md`](reports/eda_report.md)。

### 本地复现

目标环境为 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m credit_risk.data download
python -m credit_risk.data audit --output reports/data_quality_report.md
python -m pytest -q
```

执行分析 Notebook 并重新生成报告和图表：

```powershell
jupyter-nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb
jupyter-nbconvert --to notebook --execute --inplace notebooks/03_model_experiments.ipynb
```

当前自动化测试套件包含 41 个通过的测试。下载命令会同时验证官方 ZIP 和内部 XLS 的 SHA-256；原始数据只保存在本地 `data/raw/`。

### 仓库结构

```text
.
|-- data/             # 数据来源、下载和许可说明
|-- notebooks/        # 可执行的数据审计、EDA 和模型实验
|-- reports/          # 双语报告与实际生成图表
|-- references/       # 字段字典和来源
|-- src/credit_risk/  # 可复用的数据、分析、建模和评估代码
|-- tests/            # 自动化测试
|-- README.md
|-- LICENSE
|-- requirements.txt
`-- .gitignore
```

### 后续阶段

后续计划为 WOE/IV 评分卡、树模型 Challenger、概率校准、假设成本下的阈值策略，以及 PSI 和 Champion-Challenger 监控设计。

### License

项目代码采用 [MIT License](LICENSE)。数据集受来源页面的 CC BY 4.0 条款约束，不因本仓库的代码许可证而改变。

---

## English

This portfolio project builds a reproducible workflow from public credit-risk data, covering data audit, probability-of-default modeling, scorecards, model comparison and calibration, approval thresholds, and hypothetical risk-return simulation.

> **Status:** Stages 1 through 4 are complete, including source verification, data-quality audit, exploratory data analysis, a prior-probability dummy baseline, and logistic-regression baselines. WOE/IV scorecards, a Challenger model, strategy simulation, and monitoring design have not yet been implemented.

### Business question

Under transparent assumptions that approving a customer who defaults causes a credit loss and rejecting a non-defaulting customer forgoes potential return, how should an approval threshold balance bad-debt risk and approval rate? All returns, losses, and approval strategies in this project are teaching assumptions, not an institution's operating parameters or lending decisions.

### Data source and boundaries

Version one uses the UCI Machine Learning Repository [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset under CC BY 4.0. It contains 30,000 records with an observed default rate of 22.12%. Raw data are not committed. Download and verification instructions are in [`data/README.md`](data/README.md), and field definitions are in [`references/data_dictionary.md`](references/data_dictionary.md).

- The public teaching and research data do not represent a current bank or consumer-finance business.
- Historical bills, payments, and repayment status support existing-customer behavioral-risk prediction, not genuine new-customer underwriting.
- The data do not support strict temporal validation; the stratified random test set is not OOT.
- Correlations and group default-rate differences are not interpreted causally.
- `customer_id` is an identifier and never enters the model.

### Completed methods

A fixed random seed of `42` creates target-stratified training, validation, and test sets containing 60%, 20%, and 20% of observations. Categorical features use `OneHotEncoder(handle_unknown="ignore")`, and numeric features use `StandardScaler`; all preprocessing is fitted only on training data. Three models are compared:

1. a `DummyClassifier(strategy="prior")` prior-probability baseline;
2. unweighted Logistic Regression as the primary interpretable baseline;
3. Logistic Regression with `class_weight="balanced"` as an imbalance-sensitivity analysis.

Evaluation uses ROC-AUC, PR-AUC, KS, Brier Score, calibration curves, and confusion matrices, Precision, and Recall at a fixed `0.5` threshold. Accuracy is not primary, and `0.5` is not a business approval threshold.

### Observed test results

| Model | ROC-AUC | PR-AUC | KS | Brier Score | Precision @ 0.5 | Recall @ 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| Prior-probability dummy | 0.5000 | 0.2212 | 0.0000 | 0.1723 | 0.0000 | 0.0000 |
| Unweighted logistic regression | 0.7591 | 0.5258 | 0.3970 | 0.1388 | 0.6662 | 0.3519 |
| Class-balanced logistic regression | 0.7590 | 0.5226 | 0.3946 | 0.1857 | 0.4848 | 0.5531 |

The unweighted model has better probability quality than the class-balanced version. At `0.5`, the weighted model improves Recall while reducing Precision and producing more false positives. This is a statistical trade-off, not a business-optimal-model claim. See [`reports/model_report.md`](reports/model_report.md) for full results and [`reports/eda_report.md`](reports/eda_report.md) for EDA.

### Local reproduction

The target environment is Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m credit_risk.data download
python -m credit_risk.data audit --output reports/data_quality_report.md
python -m pytest -q
```

Execute the analysis notebooks to regenerate reports and figures:

```powershell
jupyter-nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb
jupyter-nbconvert --to notebook --execute --inplace notebooks/03_model_experiments.ipynb
```

The current automated suite contains 41 passing tests. The download command verifies SHA-256 for both the official ZIP and inner XLS. Raw data remain local under `data/raw/`.

### Repository structure

```text
.
|-- data/             # Data source, download, and licensing notes
|-- notebooks/        # Executable audit, EDA, and model experiments
|-- reports/          # Bilingual reports and generated figures
|-- references/       # Field dictionary and sources
|-- src/credit_risk/  # Reusable data, analysis, modeling, and evaluation code
|-- tests/            # Automated tests
|-- README.md
|-- LICENSE
|-- requirements.txt
`-- .gitignore
```

### Next stages

Planned work includes a WOE/IV scorecard, a tree-model Challenger, probability calibration, threshold strategy under hypothetical costs, and PSI plus Champion-Challenger monitoring design.

### License

Project code uses the [MIT License](LICENSE). The dataset remains subject to the source page's CC BY 4.0 terms and is unaffected by the repository's code license.
