# 测试说明（中文）

自动化测试覆盖数据下载与模式校验、可复现分层切分、特征边界、预处理 Pipeline、Dummy 和逻辑回归模型、分类评估指标、EDA 汇总、双语报告及图表生成。当前套件包含 41 个测试。

```powershell
python -m pytest -q
```

真实原始数据只用于经过校验的数据集成测试；建模、评估和报告单元测试使用可重复的合成数据，不依赖网络。

---

# Test Guide (English)

Automated tests cover data download and schema validation, reproducible stratified splitting, feature boundaries, preprocessing pipelines, Dummy and logistic models, classification metrics, EDA summaries, bilingual reports, and figure generation. The current suite contains 41 tests.

```powershell
python -m pytest -q
```

Verified raw data are used only by the data integration test. Modeling, evaluation, and report unit tests use reproducible synthetic data and require no network access.
