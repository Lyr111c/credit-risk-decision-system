# 阶段三与阶段四实施计划（中文）

> 对应规格：`docs/superpowers/specs/2026-07-28-eda-logistic-baseline-design.md`

## 实施原则

- 在 `.worktrees/eda-logistic-baseline` 隔离工作树和 `codex/eda-logistic-baseline` 分支实施。
- 使用项目根目录 `.venv` 中的 Python 3.11；原始数据从主工作区被忽略的 `data/raw/` 读取或链接，不提交数据文件。
- 每个行为先编写失败测试，确认失败原因正确，再写最小实现并运行相关测试。
- 所有报告和图表由实际执行生成；中英双语报告中文在前，英文在后且事实一致。
- 测试集在模型规则冻结前不用于选择，最终生成模型报告时评估一次。

## 任务 1：建立隔离环境并验证基线

涉及文件：无业务文件修改。

步骤：

1. 确认 `.worktrees/` 已被 Git 忽略。
2. 从当前 `main` 创建 `.worktrees/eda-logistic-baseline` 和 `codex/eda-logistic-baseline`。
3. 在工作树中运行 `D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m pytest -q`。
4. 确认 15 个既有测试通过，并确认数据文件可由明确路径读取。

## 任务 2：实现可复现数据切分和建模 Pipeline

文件：

- 新增 `tests/test_modeling.py`；
- 新增 `src/credit_risk/modeling.py`。

测试先覆盖：

1. 特征分组完整且互斥，排除 `customer_id` 和目标列。
2. 60%/20%/20% 分层切分互斥、完整、可复现。
3. 非法输入和缺失列抛出清晰错误。
4. `ColumnTransformer` 对类别变量独热编码，对数值变量标准化，并可处理未知类别。
5. 三个候选模型均为可拟合 Pipeline，并使用相同特征边界。

最小实现包括固定配置常量、`DataSplits` 数据类、切分函数、预处理器构造函数，以及 Dummy、无权重 Logistic Regression、加权 Logistic Regression 的构造函数。

验证命令：

```powershell
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m pytest tests/test_modeling.py -q
```

## 任务 3：实现统一分类评估

文件：

- 新增 `tests/test_evaluation.py`；
- 新增 `src/credit_risk/evaluation.py`。

测试先使用可手算的小样本覆盖：ROC-AUC、Average Precision、KS、Brier Score、阈值 `0.5` 下的混淆矩阵、Precision、Recall，以及校准分箱点。还要覆盖目标非二元、长度不一致和概率越界错误。

实现返回结构化、可序列化的评估结果，不依赖 Notebook 全局状态。验证命令：

```powershell
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q
```

## 任务 4：实现 EDA 汇总和切分分布检查

文件：

- 新增 `tests/test_analysis.py`；
- 新增 `src/credit_risk/analysis.py`。

测试先覆盖：

1. 目标汇总、数值分位数、零值比例、负值比例和偏度。
2. 类别频数和组内违约率。
3. 数值标准化均值差和类别最大比例差。
4. 双语 EDA 报告顺序及两个版本关键数字一致性。
5. 图表函数在临时目录生成预期文件且不依赖交互式后端。

实现使用 pandas、matplotlib 和 seaborn，限制图表数量并在图题中同时提供中英文。报告生成函数接收实际汇总结果，不嵌入预设数据指标。

## 任务 5：实现端到端基线实验和模型报告

文件：

- 新增 `tests/test_experiment.py`；
- 新增 `src/credit_risk/experiment.py`。

测试先使用合成数据验证：

1. 三个模型共享相同切分。
2. 训练只调用训练集，验证和测试只用于预测。
3. 输出明确区分 validation 和 test。
4. 双语报告包含相同模型、指标、阈值与限制。

实现提供命令行入口，加载已校验数据，冻结切分与模型配置，拟合三个模型，生成 ROC/PR、校准和混淆矩阵图，并写出模型报告。测试集调用集中在最终报告路径，避免探索函数无意重复访问。

## 任务 6：运行真实 EDA 并生成阶段三产物

文件：

- 新增 `notebooks/02_eda.ipynb`；
- 生成 `reports/eda_report.md`；
- 生成阶段三所需 `reports/figures/*.png`。

步骤：

1. 使用校验过的 UCI XLS 运行 EDA 入口。
2. 生成报告和有限数量的决策相关图表。
3. 创建调用 `src/` 逻辑的 Notebook，并从头执行保存输出。
4. 核对报告数字来自实际数据，相关性结论不含因果措辞，随机切分不称为漂移或 OOT。

## 任务 7：运行真实基线实验并生成阶段四产物

文件：

- 新增 `notebooks/03_model_experiments.ipynb`；
- 生成 `reports/model_report.md`；
- 生成阶段四所需 `reports/figures/*.png`。

步骤：

1. 在训练集拟合三个冻结模型并在验证集比较。
2. 使用固定 `0.5` 阈值；不进行阈值优化或超参数搜索。
3. 对冻结候选和 Dummy 基线执行一次测试集评估。
4. 生成双语报告和模型图表。
5. 创建调用 `src/` 逻辑的 Notebook，并从头执行保存输出。

## 任务 8：更新项目入口文档

文件：

- 实质性更新 `README.md`、`notebooks/README.md`、`reports/README.md`。

更新运行命令、阶段完成状态、实际结果摘要和产物索引。所有面向读者的更新提供完整中文和英文版本，避免把已有客户风险预测描述为新客户审批或真实经营结果。

## 任务 9：完成验证、审查和 Git 交付

验证：

```powershell
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m pytest -q
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/02_eda.ipynb --output 02_eda.ipynb --output-dir notebooks
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/03_model_experiments.ipynb --output 03_model_experiments.ipynb --output-dir notebooks
git diff --check
git status --short
```

随后进行需求逐项审查和代码审查，确认没有原始数据、缓存或临时文件进入提交。将实现提交到 `codex/eda-logistic-baseline`，推送远端并根据完成分支流程决定合并方式。

---

# Stage 3 and Stage 4 Implementation Plan (English)

> Source specification: `docs/superpowers/specs/2026-07-28-eda-logistic-baseline-design.md`

## Implementation principles

- Implement in the isolated `.worktrees/eda-logistic-baseline` worktree on branch `codex/eda-logistic-baseline`.
- Use Python 3.11 from the project-root `.venv`. Read raw data from the ignored main-workspace `data/raw/` through an explicit path or link; never commit data files.
- Write a failing test for each behavior, confirm that it fails for the expected reason, then add the minimum implementation and run the focused test.
- Generate every report and figure through actual execution. Bilingual reports place Chinese before English and contain identical facts.
- Do not use test data for selection before model rules are frozen. Evaluate it once when producing the final model report.

## Task 1: Create the isolated environment and verify the baseline

Files: no business-file changes.

Steps:

1. Confirm that Git ignores `.worktrees/`.
2. Create `.worktrees/eda-logistic-baseline` and `codex/eda-logistic-baseline` from current `main`.
3. Run `D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m pytest -q` in the worktree.
4. Confirm that all 15 existing tests pass and that an explicit path can read the dataset.

## Task 2: Implement reproducible splitting and modeling pipelines

Files:

- add `tests/test_modeling.py`;
- add `src/credit_risk/modeling.py`.

Tests first cover complete and disjoint feature groups; exclusion of `customer_id` and the target; mutually exclusive, complete, reproducible, stratified 60%/20%/20% splits; clear errors for invalid input; correct categorical and numeric preprocessing with unseen-category support; and fit-ready pipelines for all three candidate models using the same feature boundary.

The minimum implementation provides centralized constants, a `DataSplits` data class, a split function, a preprocessor builder, and builders for Dummy, unweighted Logistic Regression, and weighted Logistic Regression.

Verification:

```powershell
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m pytest tests/test_modeling.py -q
```

## Task 3: Implement unified classification evaluation

Files:

- add `tests/test_evaluation.py`;
- add `src/credit_risk/evaluation.py`.

Tests first use hand-checkable examples for ROC-AUC, Average Precision, KS, Brier Score, the confusion matrix, Precision, and Recall at `0.5`, and calibration-bin points. They also cover non-binary targets, mismatched lengths, and probabilities outside `[0, 1]`.

The implementation returns structured, serializable results without notebook-global state.

## Task 4: Implement EDA summaries and split-distribution checks

Files:

- add `tests/test_analysis.py`;
- add `src/credit_risk/analysis.py`.

Tests first cover target summaries; numeric quantiles, zero shares, negative shares, and skewness; category frequencies and within-group default rates; numeric standardized mean differences; categorical maximum proportion differences; Chinese-before-English report structure with consistent key figures; and expected figure creation in a temporary directory using a non-interactive backend.

The implementation uses pandas, matplotlib, and seaborn, limits chart count, and includes Chinese and English in figure titles. Report generation consumes observed summaries and never embeds preset data metrics.

## Task 5: Implement the end-to-end baseline experiment and model report

Files:

- add `tests/test_experiment.py`;
- add `src/credit_risk/experiment.py`.

Tests first use synthetic data to verify shared splits across models, training-only fitting, prediction-only validation and test paths, explicit validation/test separation, and consistent bilingual model names, metrics, thresholds, and limitations.

The command-line entry point loads verified data, freezes the split and model configuration, fits all three models, generates ROC/PR, calibration, and confusion-matrix figures, and writes the model report. Test-set access remains centralized in the final-report path so exploratory functions cannot repeatedly inspect it accidentally.

## Task 6: Run real EDA and generate Stage 3 artifacts

Files:

- add `notebooks/02_eda.ipynb`;
- generate `reports/eda_report.md`;
- generate required Stage 3 `reports/figures/*.png`.

Run the EDA entry point against the verified UCI workbook, generate a limited decision-relevant figure set and report, create a notebook that calls `src/` logic, and execute it from start to finish. Verify that all numbers come from observed data, correlation language is non-causal, and random-split differences are not labeled drift or OOT.

## Task 7: Run the real baseline experiment and generate Stage 4 artifacts

Files:

- add `notebooks/03_model_experiments.ipynb`;
- generate `reports/model_report.md`;
- generate required Stage 4 `reports/figures/*.png`.

Fit the three frozen models on training data, compare them on validation data, use the fixed `0.5` threshold without threshold optimization or hyperparameter search, evaluate the frozen candidates and Dummy baseline once on test data, generate the bilingual report and figures, and execute the notebook from start to finish.

## Task 8: Update project entry documentation

Files:

- materially update `README.md`, `notebooks/README.md`, and `reports/README.md`.

Add execution commands, stage status, observed-result summaries, and artifact links. Every reader-facing update contains complete Chinese and English versions and avoids describing existing-customer risk prediction as new-customer underwriting or actual operating results.

## Task 9: Complete verification, review, and Git delivery

Run:

```powershell
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m pytest -q
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/02_eda.ipynb --output 02_eda.ipynb --output-dir notebooks
D:\Credit_Risk_Decision_System\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/03_model_experiments.ipynb --output 03_model_experiments.ipynb --output-dir notebooks
git diff --check
git status --short
```

Then perform requirement-by-requirement and code reviews, confirm that raw data, caches, and temporary files are absent from the commit, commit the implementation on `codex/eda-logistic-baseline`, push it, and use the development-branch completion workflow to choose integration.
