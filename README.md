# Credit Risk Decision System

基于公开信用风险数据，构建从数据审计、违约概率建模、信用评分卡、模型比较与概率校准，到审批阈值和假设风险收益模拟的可复现流程。

> **Status:** 项目初始化阶段。当前仓库仅包含项目结构、数据来源说明和依赖定义，尚未产出模型结果。

## English Summary

This portfolio project develops a reproducible credit-risk decision workflow using public data. It will cover data auditing, probability-of-default modeling, scorecards, model calibration, approval-threshold analysis, and scenario-based decision simulation.

## 业务问题

在假设误批违约客户会产生信用损失、误拒优质客户会损失潜在收益的条件下，如何选择审批阈值，在控制坏账风险的同时维持合理通过率？

本项目中的收益、损失和审批策略均为教学用途的透明假设，不代表真实金融机构的经营参数或授信决策。

## 数据来源

第一版计划使用 UCI Machine Learning Repository 的 [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) 数据集。原始数据默认不提交到仓库；下载方式、字段说明和使用边界记录在 [`data/README.md`](data/README.md)。

## 第一版范围

- 数据字典、数据质量检查与探索性分析；
- 逻辑回归基线与 WOE/IV 信用评分卡；
- 一个树模型作为 Challenger；
- ROC-AUC、PR-AUC、KS、Brier Score 和概率校准评估；
- 审批阈值、通过率、坏账率及假设成本情景分析；
- PSI、数据漂移与 Champion-Challenger 监控设计。

深度学习、实时系统、微服务、复杂前端和云部署不在第一版范围内。

## 数据与业务限制

- 数据来自公开教学与研究场景，不代表当前中国银行或消费金融业务；
- 数据不能完整还原真实申请审批流程；
- 数据不支持严格的跨时期样本验证，随机切分不会被描述为 OOT 验证；
- 项目不能可靠处理拒绝推断、宏观周期或真实资金成本；
- 模型相关性和特征重要性不会被解释为因果影响。

## 计划流程

1. 数据来源与字段核验；
2. 数据质量审计与 EDA；
3. 逻辑回归和评分卡基线；
4. Challenger 模型与概率校准；
5. 阈值及假设成本情景分析；
6. 稳定性监控设计、测试和模型报告。

## 本地环境

目标环境为 Python 3.11。后续安装方式：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

当前阶段尚未验证依赖安装；完成 Python 3.11 环境配置后再执行验证。

## 仓库结构

```text
.
|-- data/             # 数据来源、下载和许可说明
|-- notebooks/        # 按分析阶段编号的探索性 Notebook
|-- reports/          # 模型报告与生成图表
|-- references/       # 数据、论文、库和参考项目来源
|-- src/credit_risk/  # 可复用的数据、建模、评价和策略代码
|-- tests/            # 自动化测试
|-- README.md
|-- LICENSE
|-- requirements.txt
`-- .gitignore
```

## License

项目代码采用 [MIT License](LICENSE)。数据集受其来源页面所列条款约束，不因本仓库的代码许可证而改变。
