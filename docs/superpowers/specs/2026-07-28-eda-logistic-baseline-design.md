# 阶段三与阶段四设计规格（中文）

## 1. 目标与范围

本阶段为 UCI Default of Credit Card Clients 数据集完成两项工作：

1. 阶段三：进行面向建模决策的探索性数据分析（EDA）。
2. 阶段四：建立总体违约率虚拟基线和可解释的逻辑回归基线。

分析对象是已有客户的行为风险，而不是真实的新客户贷前审批。数据不支持真正的样本外时间验证，因此所有切分均称为分层随机切分，不称为 OOT。阶段三和阶段四不包含 WOE/IV、评分卡、树模型、概率再校准、业务成本阈值优化或 PSI 监控；这些内容留给后续阶段。

## 2. 设计选择

采用模块化可复现实验方案：Notebook 负责解释和展示，可复用的数据切分、预处理、建模、评估与报告逻辑放入 `src/credit_risk/`，关键行为由自动化测试覆盖。该方案比 Notebook 集中实现更容易验证和复用，同时避免在简单基线阶段引入完整调参框架。

固定随机种子为 `42`。数据按目标变量分层切分为：

- 训练集：60%，约 18,000 条；
- 验证集：20%，约 6,000 条；
- 测试集：20%，约 6,000 条。

先从完整数据中保留 20% 测试集，再把剩余数据按 75%/25% 切为训练集和验证集，从而得到最终 60%/20%/20% 比例。测试集在特征集合、预处理、模型配置、模型比较规则和阈值规则确定前不参与选择；最终报告阶段只进行一次测试集评估。

## 3. 数据与特征边界

目标变量为 `default_next_month`，正类 `1` 表示下个月违约。`customer_id` 只作为记录标识，必须从模型特征中排除。

类别特征为：

- `sex`；
- `education`；
- `marital_status`；
- 六个 `repayment_status_*` 字段。

数值特征为：

- `credit_limit`；
- `age`；
- 六个 `bill_amount_*` 字段；
- 六个 `payment_amount_*` 字段。

未在 UCI 文档中定义的类别编码继续保留为独立类别，不擅自合并或解释。负账单金额和极端金额继续保留；EDA 会明确展示其存在，基线阶段不依据未经证实的业务含义删除或截尾。

## 4. 阶段三：EDA 设计

EDA 只生成支持后续建模判断的紧凑结果，不生成大规模自动化画像报告。内容包括：

1. 目标类别计数、比例和总体违约率，说明 Accuracy 不能作为主要指标。
2. 数值特征的分位数、零值比例、负值比例和偏度；对额度、年龄、账单和还款金额使用有限数量的代表性分布图。
3. 类别特征的频数和组内违约率，保留并标注来源未定义编码。
4. 六个月还款状态的频率及违约率模式，只描述关联，不作因果解释。
5. 数值特征相关矩阵和与目标的点二列相关性，仅用于识别冗余与候选关系，不解释为因果效应。
6. 切分后的目标比例检查；数值特征使用标准化均值差，类别特征使用最大类别比例差检查训练、验证和测试分布。检查结果用于描述随机切分的相似程度，不把随机波动包装为显著业务漂移。

主要图表保存到 `reports/figures/`，图题和图注同时包含中文与英文。`notebooks/02_eda.ipynb` 展示分析过程，`reports/eda_report.md` 提供完整的中文部分和对应英文部分；两个语言版本使用相同数字、结论、限制和警告。

## 5. 阶段四：建模设计

使用 scikit-learn `Pipeline` 和 `ColumnTransformer`，确保预处理只在训练数据上拟合：

- 类别特征使用 `OneHotEncoder(handle_unknown="ignore")`；
- 数值特征使用 `StandardScaler()`；
- 逻辑回归使用固定随机种子和足够的最大迭代次数，避免无意的收敛截断。

在完全相同的数据切分上比较：

1. `DummyClassifier(strategy="prior")`：以训练集总体违约率输出常数概率，作为最低概率基线。
2. 无类别权重的 Logistic Regression：主要可解释基线。
3. `class_weight="balanced"` 的 Logistic Regression：作为类别不平衡敏感性分析，不替代主要基线。

不进行超参数搜索。验证集用于比较模型，并在候选模型之间按以下优先顺序作出选择：ROC-AUC、PR-AUC、Brier Score 与校准表现，同时记录 KS；不会仅以 Accuracy 或单个阈值指标选择模型。如果无权重逻辑回归与加权版本各有优劣，报告保留二者，不宣称存在无条件胜者。

分类阈值固定为 `0.5`，用于生成混淆矩阵、Precision 和 Recall。该阈值不是业务审批阈值，也不宣称最优；业务成本驱动的阈值选择属于后续策略阶段。

## 6. 评估定义

每个数据集与模型使用一致的正类定义和评估函数，输出：

- ROC-AUC；
- PR-AUC（Average Precision）；
- KS，定义为 ROC 曲线上 `max(TPR - FPR)`；
- Brier Score；
- `0.5` 阈值下的混淆矩阵、Precision 和 Recall；
- 分箱校准曲线，使用固定分箱数并返回各箱平均预测概率与实际违约率。

测试集只用于最终候选模型和虚拟基线的冻结评估。报告必须清楚区分验证集结果与测试集结果，且不把随机测试集称为 OOT。

## 7. 代码与产物结构

计划新增或实质性更新：

- `src/credit_risk/modeling.py`：特征分组、分层切分、预处理器和模型 Pipeline 构建。
- `src/credit_risk/evaluation.py`：指标、KS、校准数据和混淆矩阵计算。
- `src/credit_risk/analysis.py`：EDA 汇总、切分分布检查、图表与双语报告生成入口。
- `src/credit_risk/experiment.py`：加载数据、训练候选模型、生成冻结指标与双语模型报告。
- `tests/test_modeling.py`、`tests/test_evaluation.py`、`tests/test_analysis.py` 和必要的报告测试。
- `notebooks/02_eda.ipynb`：可执行的 EDA 展示。
- `notebooks/03_model_experiments.ipynb`：可执行的基线模型展示。
- `reports/eda_report.md`、`reports/model_report.md`：由代码生成的双语报告。
- `reports/figures/`：EDA、ROC/PR、校准和混淆矩阵等必要图表。

Notebook 不承载唯一实现。报告由代码生成；任何双语内容修改必须同步修改生成逻辑和测试，再重新生成产物。

## 8. 错误处理与可复现性

公共函数在输入缺少目标列、特征列不完整、切分比例非法、概率长度不一致或目标不是二元值时抛出清晰的 `ValueError`。数据加载继续复用现有校验和与模式校验逻辑。

随机种子、切分比例、阈值、校准分箱数和模型参数集中定义并在报告中记录。所有实际指标与图表必须由本地已校验数据运行生成，不手工填写或估算。原始数据继续留在被忽略的 `data/raw/` 中，不提交到 Git。

## 9. 测试与验收标准

实施采用测试驱动方式，至少验证：

1. 60%/20%/20% 切分的样本互斥、覆盖完整、可复现且目标比例接近。
2. `customer_id` 和目标列不会进入模型输入。
3. 类别与数值预处理符合定义，未知类别可以转换。
4. 预处理器只从训练集拟合，验证集和测试集仅调用转换与预测路径。
5. ROC-AUC、PR-AUC、KS、Brier Score、混淆矩阵、Precision、Recall 和校准点对已知小样本给出正确结果。
6. 虚拟基线、无权重逻辑回归和加权逻辑回归使用相同切分。
7. 报告中文在前、英文在后，两个版本包含一致的关键数字与限制。
8. 两个 Notebook 能从头执行，且生成的报告和图表可重复生成。
9. 完整测试套件通过，Git 状态不包含原始数据、缓存或临时文件。

完成阶段三和阶段四时，必须有实际运行得到的 EDA 结果、验证集与测试集模型指标、图表和报告；仅有函数或空 Notebook 不满足验收标准。

---

# Stage 3 and Stage 4 Design Specification (English)

## 1. Objective and scope

This stage completes two bodies of work for the UCI Default of Credit Card Clients dataset:

1. Stage 3: exploratory data analysis (EDA) focused on modeling decisions.
2. Stage 4: an overall-default-rate dummy baseline and an interpretable logistic-regression baseline.

The analysis concerns behavioral risk for existing customers, not genuine new-customer underwriting. The data do not support true out-of-time validation, so every split is described as a stratified random split and never as OOT. Stages 3 and 4 exclude WOE/IV, scorecards, tree models, probability recalibration, business-cost threshold optimization, and PSI monitoring; those belong to later stages.

## 2. Design choice

The implementation uses a modular, reproducible experiment design. Notebooks explain and display results, while reusable splitting, preprocessing, modeling, evaluation, and report logic resides in `src/credit_risk/` and is covered by automated tests. This is easier to verify and reuse than a notebook-centered implementation and avoids introducing a full tuning framework during the simple-baseline stage.

The fixed random seed is `42`. The target-stratified split is:

- training: 60%, approximately 18,000 observations;
- validation: 20%, approximately 6,000 observations;
- test: 20%, approximately 6,000 observations.

First, 20% of the full dataset is held out as test data. The remaining data are then divided 75%/25% into training and validation data, producing the final 60%/20%/20% proportions. The test set does not participate in feature-set, preprocessing, model-configuration, model-comparison-rule, or threshold-rule selection. It is evaluated once during final reporting.

## 3. Data and feature boundaries

The target is `default_next_month`, where positive class `1` means default in the following month. `customer_id` is only a record identifier and must be excluded from model features.

Categorical features are:

- `sex`;
- `education`;
- `marital_status`;
- the six `repayment_status_*` fields.

Numeric features are:

- `credit_limit`;
- `age`;
- the six `bill_amount_*` fields;
- the six `payment_amount_*` fields.

Category codes not defined in the UCI documentation remain separate categories and are not merged or interpreted without evidence. Negative bill amounts and extreme amounts are retained. EDA documents their presence, while the baseline does not remove or winsorize them based on unverified business meanings.

## 4. Stage 3: EDA design

EDA produces a compact set of results that support later modeling decisions rather than a large automated profiling report. It includes:

1. Target counts, proportions, and overall default rate, with an explanation of why Accuracy is not a primary metric.
2. Numeric-feature quantiles, zero shares, negative shares, and skewness, plus a limited set of representative distribution plots for credit limit, age, bill amounts, and payment amounts.
3. Categorical-feature frequencies and within-group default rates, retaining and labeling source-undefined codes.
4. Six-month repayment-status frequency and default-rate patterns, described as associations without causal claims.
5. A numeric-feature correlation matrix and point-biserial correlations with the target, used only to identify redundancy and candidate relationships, not causal effects.
6. Post-split target-rate checks; standardized mean differences for numeric features and maximum category-proportion differences for categorical features across training, validation, and test data. These checks describe random-split similarity and do not present random variation as meaningful business drift.

Primary figures are saved under `reports/figures/`, with bilingual titles and captions. `notebooks/02_eda.ipynb` presents the analysis, while `reports/eda_report.md` contains a complete Chinese section followed by a corresponding English section. Both language versions use identical figures, conclusions, limitations, and warnings.

## 5. Stage 4: modeling design

The implementation uses scikit-learn `Pipeline` and `ColumnTransformer` so that preprocessing is fitted only on training data:

- categorical features use `OneHotEncoder(handle_unknown="ignore")`;
- numeric features use `StandardScaler()`;
- logistic regression uses the fixed random seed and a sufficiently high iteration limit to avoid accidental convergence truncation.

The same data split is used to compare:

1. `DummyClassifier(strategy="prior")`: emits the training-set overall default probability as a constant-probability floor baseline.
2. Logistic Regression without class weights: the primary interpretable baseline.
3. Logistic Regression with `class_weight="balanced"`: an imbalance-sensitivity analysis, not a replacement for the primary baseline.

No hyperparameter search is performed. The validation set supports model comparison using, in priority order, ROC-AUC, PR-AUC, Brier Score, and calibration behavior, while also recording KS. Accuracy or a single threshold metric is never the sole selection criterion. If unweighted and weighted logistic regression have different strengths, the report retains both and does not claim an unconditional winner.

The classification threshold is fixed at `0.5` for confusion matrices, Precision, and Recall. It is neither a business approval threshold nor claimed to be optimal. Business-cost-driven threshold selection belongs to the later strategy stage.

## 6. Evaluation definitions

Each dataset and model uses the same positive-class definition and evaluation function, producing:

- ROC-AUC;
- PR-AUC (Average Precision);
- KS, defined as `max(TPR - FPR)` on the ROC curve;
- Brier Score;
- confusion matrix, Precision, and Recall at threshold `0.5`;
- a binned calibration curve that uses a fixed bin count and returns mean predicted probability and observed default rate per bin.

The test set is used only for the frozen evaluation of the final candidate models and dummy baseline. Reports clearly distinguish validation results from test results and never describe the random test set as OOT.

## 7. Code and artifact structure

Planned additions or material updates are:

- `src/credit_risk/modeling.py`: feature groups, stratified splitting, preprocessing, and model-pipeline construction.
- `src/credit_risk/evaluation.py`: metrics, KS, calibration data, and confusion-matrix computation.
- `src/credit_risk/analysis.py`: EDA summaries, split-distribution checks, figures, and bilingual-report generation entry point.
- `src/credit_risk/experiment.py`: data loading, candidate-model fitting, frozen metrics, and bilingual-model-report generation.
- `tests/test_modeling.py`, `tests/test_evaluation.py`, `tests/test_analysis.py`, and necessary report tests.
- `notebooks/02_eda.ipynb`: executable EDA presentation.
- `notebooks/03_model_experiments.ipynb`: executable baseline-model presentation.
- `reports/eda_report.md` and `reports/model_report.md`: code-generated bilingual reports.
- `reports/figures/`: necessary EDA, ROC/PR, calibration, and confusion-matrix figures.

Notebooks do not contain the only implementation. Reports are code-generated. Any bilingual-content change must update the generator and tests before regenerating artifacts.

## 8. Error handling and reproducibility

Public functions raise clear `ValueError` exceptions for a missing target, incomplete feature columns, invalid split proportions, mismatched probability lengths, or a non-binary target. Data loading continues to reuse the existing checksum and schema validation.

The random seed, split proportions, threshold, calibration-bin count, and model parameters are centralized and recorded in reports. Every reported metric and figure is generated by executing the verified local data; none is entered manually or estimated. Raw data remain under ignored `data/raw/` and are never committed to Git.

## 9. Tests and acceptance criteria

Implementation follows test-driven development and verifies at least:

1. The 60%/20%/20% splits are mutually exclusive, collectively complete, reproducible, and close in target rate.
2. `customer_id` and the target do not enter model input.
3. Categorical and numeric preprocessing match the design, including transformation of unseen categories.
4. The preprocessor is fitted only on training data; validation and test data follow transform and prediction paths only.
5. ROC-AUC, PR-AUC, KS, Brier Score, confusion matrix, Precision, Recall, and calibration points are correct for known small examples.
6. The dummy baseline, unweighted logistic regression, and weighted logistic regression use the same split.
7. Reports place Chinese before English and contain consistent key figures and limitations in both versions.
8. Both notebooks execute from start to finish and regenerate their reports and figures.
9. The full test suite passes, and Git status excludes raw data, caches, and temporary files.

Completion of Stages 3 and 4 requires EDA results, validation and test model metrics, figures, and reports produced by actual execution. Functions alone or empty notebooks do not satisfy acceptance criteria.
