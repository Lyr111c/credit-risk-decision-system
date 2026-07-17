# Data

## Planned dataset

- Name: Default of Credit Card Clients
- Provider: UCI Machine Learning Repository
- Dataset page: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- DOI: <https://doi.org/10.24432/C55S3H>

## Local data policy

原始数据不直接提交到本仓库。下载前应在 UCI 数据集页面重新核对许可、引用要求、文件内容和字段说明。数据文件只保存在本地 `data/` 子目录中，并由根目录 `.gitignore` 排除。

建议后续使用以下本地目录：

```text
data/
|-- raw/          # 保持下载后的原始内容不变
|-- interim/      # 数据审计或清洗的中间结果
`-- processed/    # 可复现流程生成的建模输入
```

## Known limitations

- 公开教学与研究数据不能代表当前中国金融机构的客户群体；
- 数据不能完整还原真实的贷前审批过程；
- 缺少可用于严格 OOT 验证的跨时期样本；
- 收益、损失和策略分析只能作为假设情景。

具体下载命令、文件校验值和字段字典将在核对数据来源后补充，并以实际执行结果为准。
