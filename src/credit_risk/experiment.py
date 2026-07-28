"""End-to-end dummy and logistic-regression baseline experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

from credit_risk.data import RAW_XLS_NAME, load_dataset
from credit_risk.evaluation import (
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_THRESHOLD,
    evaluate_binary_classifier,
)
from credit_risk.modeling import (
    LOGISTIC_MAX_ITER,
    DataSplits,
    build_model_pipelines,
    split_dataset,
)

MODEL_NAMES_ZH = {
    "dummy": "总体违约率虚拟基线",
    "logistic": "无权重逻辑回归",
    "logistic_balanced": "类别平衡逻辑回归",
}
MODEL_NAMES_EN = {
    "dummy": "Prior-probability dummy",
    "logistic": "Unweighted logistic regression",
    "logistic_balanced": "Class-balanced logistic regression",
}
FIGURE_NAMES = (
    "model_roc_curves.png",
    "model_pr_curves.png",
    "model_calibration.png",
    "model_confusion_matrices.png",
)


@dataclass(frozen=True)
class ExperimentResult:
    """Fitted models, one shared split, predictions, and frozen metrics."""

    splits: DataSplits
    models: dict[str, Pipeline]
    validation_probabilities: dict[str, np.ndarray]
    test_probabilities: dict[str, np.ndarray]
    metrics: dict[str, dict[str, dict[str, object]]]


def run_baseline_experiment(frame: pd.DataFrame) -> ExperimentResult:
    """Fit all candidates on one training split and evaluate frozen holdouts."""
    splits = split_dataset(frame)
    models = build_model_pipelines()
    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, dict[str, object]]] = {
        "validation": {},
        "test": {},
    }
    for name, model in models.items():
        model.fit(splits.X_train, splits.y_train)
        validation_probability = model.predict_proba(splits.X_validation)[:, 1]
        test_probability = model.predict_proba(splits.X_test)[:, 1]
        validation_probabilities[name] = validation_probability
        test_probabilities[name] = test_probability
        metrics["validation"][name] = evaluate_binary_classifier(
            splits.y_validation, validation_probability
        )
        metrics["test"][name] = evaluate_binary_classifier(splits.y_test, test_probability)
    return ExperimentResult(
        splits=splits,
        models=models,
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        metrics=metrics,
    )


def _configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_model_figures(
    result: ExperimentResult, output_directory: str | Path
) -> list[Path]:
    """Generate frozen test-set discrimination, calibration, and error figures."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _configure_plotting()
    paths: list[Path] = []
    y_test = result.splits.y_test

    figure, axis = plt.subplots(figsize=(7, 5.5))
    for name, probability in result.test_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
        auc = result.metrics["test"][name]["roc_auc"]
        axis.plot(false_positive_rate, true_positive_rate, label=f"{MODEL_NAMES_EN[name]} ({auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random / 随机")
    axis.set(xlabel="False-positive rate / 假阳性率", ylabel="True-positive rate / 真阳性率")
    axis.set_title("Test ROC curves / 测试集 ROC 曲线")
    axis.legend(fontsize=8)
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[0]))

    figure, axis = plt.subplots(figsize=(7, 5.5))
    for name, probability in result.test_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probability)
        auc = result.metrics["test"][name]["pr_auc"]
        axis.step(
            recall,
            precision,
            where="post",
            label=f"{MODEL_NAMES_EN[name]} ({auc:.3f})",
        )
    axis.axhline(y_test.mean(), linestyle="--", color="grey", label="Default rate / 违约率")
    axis.set(xlabel="Recall / 召回率", ylabel="Precision / 精确率")
    axis.set_title("Test precision-recall curves / 测试集 PR 曲线")
    axis.legend(fontsize=8)
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[1]))

    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect / 完美校准")
    for name in result.models:
        calibration = result.metrics["test"][name]["calibration"]
        axis.plot(
            [point["mean_predicted_probability"] for point in calibration],
            [point["observed_default_rate"] for point in calibration],
            marker="o",
            label=MODEL_NAMES_EN[name],
        )
    axis.set(
        xlabel="Mean predicted probability / 平均预测概率",
        ylabel="Observed default rate / 实际违约率",
    )
    axis.set_title("Test calibration / 测试集校准")
    axis.legend(fontsize=8)
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[2]))

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, name in zip(axes, result.models, strict=True):
        confusion = result.metrics["test"][name]["confusion_matrix"]
        matrix = np.array(
            [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
        )
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color=color,
                )
        axis.set_xticks([0, 1], ["Pred 0 / 预测 0", "Pred 1 / 预测 1"])
        axis.set_yticks([0, 1], ["Actual 0 / 实际 0", "Actual 1 / 实际 1"])
        axis.set_title(MODEL_NAMES_EN[name], fontsize=9)
    figure.suptitle("Test confusion matrices at 0.5 / 测试集 0.5 阈值混淆矩阵")
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[3]))
    return paths


def _markdown_rows(rows: list[list[str]], headers: list[str]) -> str:
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def _metrics_table(result: ExperimentResult, split: str, *, english: bool) -> str:
    headers = (
        ["Model", "ROC-AUC", "PR-AUC", "KS", "Brier Score", "Precision @ 0.5", "Recall @ 0.5"]
        if english
        else ["模型", "ROC-AUC", "PR-AUC", "KS", "Brier Score", "Precision @ 0.5", "Recall @ 0.5"]
    )
    names = MODEL_NAMES_EN if english else MODEL_NAMES_ZH
    rows = []
    for name, metrics in result.metrics[split].items():
        rows.append(
            [
                names[name],
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['pr_auc']:.4f}",
                f"{metrics['ks']:.4f}",
                f"{metrics['brier_score']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
            ]
        )
    return _markdown_rows(rows, headers)


def _confusion_table(result: ExperimentResult, *, english: bool) -> str:
    headers = (
        ["Model", "TN", "FP", "FN", "TP"]
        if english
        else ["模型", "TN", "FP", "FN", "TP"]
    )
    names = MODEL_NAMES_EN if english else MODEL_NAMES_ZH
    rows = []
    for name, metrics in result.metrics["test"].items():
        confusion = metrics["confusion_matrix"]
        rows.append(
            [
                names[name],
                f"{confusion['tn']:,}",
                f"{confusion['fp']:,}",
                f"{confusion['fn']:,}",
                f"{confusion['tp']:,}",
            ]
        )
    return _markdown_rows(rows, headers)


def format_model_report(result: ExperimentResult) -> str:
    """Render validation and frozen test results as a bilingual report."""
    splits = result.splits
    return f"""# 逻辑回归基线模型报告（中文）

本报告由 `credit_risk.experiment` 实际运行生成。数据使用固定随机种子 `42` 进行 60%/20%/20% 分层随机切分：训练集 {len(splits.X_train):,} 条、验证集 {len(splits.X_validation):,} 条、测试集 {len(splits.X_test):,} 条。该测试集不是 OOT。

## 方法

`customer_id` 被排除。类别变量在训练集内独热编码，数值变量在训练集内标准化。比较总体违约率虚拟基线、无权重逻辑回归和类别平衡逻辑回归；没有进行超参数搜索。无权重模型是主要可解释基线，加权模型只用于类别不平衡敏感性分析。

冻结参数为：`random_state=42`、`max_iter={LOGISTIC_MAX_ITER}`、{DEFAULT_CALIBRATION_BINS} 个等宽概率校准箱和分类阈值 `{DEFAULT_THRESHOLD}`。Dummy 使用 `strategy="prior"`；敏感性模型使用 `class_weight="balanced"`。

## 验证集结果

{_metrics_table(result, 'validation', english=False)}

验证集用于比较固定模型。ROC-AUC 和 PR-AUC 衡量排序，KS 衡量分离，Brier Score 和校准图衡量概率质量；Accuracy 不作为主要指标。

校准图使用等宽概率校准箱。尾部校准箱可能包含较少样本，因此单个尾部点的波动不应被过度解释。

## 测试集冻结结果

{_metrics_table(result, 'test', english=False)}

![ROC 曲线](figures/model_roc_curves.png)

![PR 曲线](figures/model_pr_curves.png)

![校准曲线](figures/model_calibration.png)

## 0.5 阈值结果

{_confusion_table(result, english=False)}

![混淆矩阵](figures/model_confusion_matrices.png)

固定 `0.5` 仅是透明的统计基线阈值，不是审批阈值，也不宣称业务最优。成本驱动阈值留给后续策略阶段。

## 结论与限制

逻辑回归结果应与虚拟基线比较，并同时考察区分度和概率质量。无权重与加权逻辑回归可能在概率校准和召回率之间体现不同取舍，因此本报告不把单一指标最高者称为无条件最优模型。数据描述已有客户行为风险，不能代表新客户贷前审批；随机测试集不是 OOT，也不能证明模型跨经济周期稳定。

---

# Logistic Regression Baseline Model Report (English)

This report was generated by executing `credit_risk.experiment`. A fixed random seed of `42` produces a 60%/20%/20% stratified random split with {len(splits.X_train):,} training, {len(splits.X_validation):,} validation, and {len(splits.X_test):,} test observations. The test set is not OOT.

## Method

`customer_id` is excluded. Categorical features are one-hot encoded and numeric features standardized using training data only. The comparison includes a prior-probability dummy, unweighted logistic regression, and class-balanced logistic regression, without hyperparameter search. The unweighted model is the primary interpretable baseline; the weighted model is only an imbalance-sensitivity analysis.

Frozen parameters are `random_state=42`, `max_iter={LOGISTIC_MAX_ITER}`, {DEFAULT_CALIBRATION_BINS} equal-width probability calibration bins, and classification threshold `{DEFAULT_THRESHOLD}`. The Dummy uses `strategy="prior"`; the sensitivity model uses `class_weight="balanced"`.

## Validation results

{_metrics_table(result, 'validation', english=True)}

Validation data compare the fixed models. ROC-AUC and PR-AUC measure ranking, KS measures separation, and Brier Score plus calibration measure probability quality. Accuracy is not a primary metric.

Calibration uses equal-width probability calibration bins. Tail bins can contain fewer observations, so variation in an individual tail point should not be overinterpreted.

## Frozen test results

{_metrics_table(result, 'test', english=True)}

![ROC curves](figures/model_roc_curves.png)

![Precision-recall curves](figures/model_pr_curves.png)

![Calibration curves](figures/model_calibration.png)

## Results at threshold 0.5

{_confusion_table(result, english=True)}

![Confusion matrices](figures/model_confusion_matrices.png)

The fixed `0.5` value is only a transparent statistical-baseline threshold. It is not an approval threshold and is not claimed to be business-optimal. Cost-driven threshold selection belongs to the later strategy stage.

## Conclusions and limitations

Logistic-regression results must be compared with the dummy baseline while considering both discrimination and probability quality. Unweighted and weighted logistic regression can express different trade-offs between calibration and recall, so this report does not call the model with the highest single metric unconditionally optimal. The data describe existing-customer behavioral risk and do not represent new-customer underwriting. The random test set is not OOT and cannot establish stability across economic cycles.
"""


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("data/raw") / RAW_XLS_NAME)
    parser.add_argument("--output", type=Path, default=Path("reports/model_report.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    result = run_baseline_experiment(load_dataset(args.path))
    generate_model_figures(result, args.figures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_model_report(result), encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    _main()
