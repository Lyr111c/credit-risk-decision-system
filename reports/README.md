# 报告说明（中文）

本目录保存经过实际运行生成的数据和模型报告。报告由 `src/credit_risk/` 中的生成逻辑产生，不手工填写指标。

- `data_quality_report.md`：30,000 条记录的数据质量、编码和可用性审计。
- `eda_report.md`：阶段三双语 EDA，包括目标不平衡、数值分布、类别违约率、还款状态、相关性和随机切分检查。
- `model_report.md`：阶段四双语模型报告，包括验证集与冻结测试集的 ROC-AUC、PR-AUC、KS、Brier Score、校准和 `0.5` 阈值结果。
- `figures/`：报告引用的六张 EDA 图和四张模型评估图。

随机测试集不是 OOT，相关性不是因果关系，`0.5` 不是业务审批阈值。

---

# Report Guide (English)

This directory stores data and model reports produced by actual execution. Generators under `src/credit_risk/` produce the reports; metrics are never entered manually.

- `data_quality_report.md`: data-quality, code, and availability audit for 30,000 records.
- `eda_report.md`: bilingual Stage 3 EDA covering target imbalance, numeric distributions, category default rates, repayment status, correlation, and random-split checks.
- `model_report.md`: bilingual Stage 4 model report covering validation and frozen-test ROC-AUC, PR-AUC, KS, Brier Score, calibration, and results at `0.5`.
- `figures/`: six EDA figures and four model-evaluation figures referenced by the reports.

The random test set is not OOT, correlation is not causation, and `0.5` is not a business approval threshold.
