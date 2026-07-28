# Notebook 说明（中文）

Notebook 仅用于探索、诊断和结果展示；可复用逻辑位于 `src/credit_risk/`。打开 Notebook 后应选择项目根目录 `.venv` 中的 Python 3.11 解释器，不要使用 Conda `base` 或其他项目环境。

- `01_data_audit.ipynb`：验证下载工作簿并展示可复现的数据质量审计。
- `02_eda.ipynb`：执行阶段三 EDA，生成 `reports/eda_report.md` 和六张 EDA 图。
- `03_model_experiments.ipynb`：执行阶段四冻结基线实验，生成 `reports/model_report.md` 和四张模型评估图。

从项目根目录重新执行：

```powershell
jupyter-nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb
jupyter-nbconvert --to notebook --execute --inplace notebooks/03_model_experiments.ipynb
```

模型 Notebook 使用固定 `60%/20%/20%` 分层随机切分。测试集不是 OOT；固定 `0.5` 阈值不是业务审批阈值。

---

# Notebook Guide (English)

Notebooks are limited to exploration, diagnostics, and presentation; reusable logic resides in `src/credit_risk/`. Select the Python 3.11 interpreter from the project-root `.venv`, not Conda `base` or another environment.

- `01_data_audit.ipynb`: verifies the downloaded workbook and presents the reproducible data-quality audit.
- `02_eda.ipynb`: runs Stage 3 EDA and generates `reports/eda_report.md` plus six EDA figures.
- `03_model_experiments.ipynb`: runs the frozen Stage 4 baseline experiment and generates `reports/model_report.md` plus four model-evaluation figures.

Re-execute from the project root:

```powershell
jupyter-nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb
jupyter-nbconvert --to notebook --execute --inplace notebooks/03_model_experiments.ipynb
```

The model notebook uses the fixed `60%/20%/20%` stratified random split. The test set is not OOT, and the fixed `0.5` threshold is not a business approval threshold.
