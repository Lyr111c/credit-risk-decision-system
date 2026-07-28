"""Decision-focused EDA summaries, figures, and bilingual reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from credit_risk.data import RAW_XLS_NAME, load_dataset
from credit_risk.modeling import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    DataSplits,
    split_dataset,
)

FIGURE_NAMES = (
    "eda_target_distribution.png",
    "eda_numeric_distributions.png",
    "eda_category_default_rates.png",
    "eda_repayment_status.png",
    "eda_correlation_heatmap.png",
    "eda_split_comparison.png",
)


def summarize_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize ranges, tails, zeros, negatives, and skew for numeric features."""
    values = frame.loc[:, NUMERIC_COLUMNS]
    return pd.DataFrame(
        {
            "minimum": values.min(),
            "p01": values.quantile(0.01),
            "median": values.median(),
            "p99": values.quantile(0.99),
            "maximum": values.max(),
            "zero_share": values.eq(0).mean(),
            "negative_share": values.lt(0).mean(),
            "skewness": values.skew(),
        }
    ).loc[list(NUMERIC_COLUMNS)]


def summarize_categorical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return observed category frequencies and within-category default rates."""
    summaries = []
    for feature in CATEGORICAL_COLUMNS:
        summary = (
            frame.groupby(feature, dropna=False)[TARGET_COLUMN]
            .agg(count="size", default_rate="mean")
            .reset_index(names="value")
        )
        summary.insert(0, "feature", feature)
        summary["share"] = summary["count"] / len(frame)
        summaries.append(summary[["feature", "value", "count", "share", "default_rate"]])
    return pd.concat(summaries, ignore_index=True)


def summarize_repayment_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Return six-month repayment-status frequency and default-rate patterns."""
    repayment_columns = [
        column for column in CATEGORICAL_COLUMNS if column.startswith("repayment_status_")
    ]
    repayment_long = frame.melt(
        id_vars=TARGET_COLUMN,
        value_vars=repayment_columns,
        var_name="month",
        value_name="status",
    )
    summary = (
        repayment_long.groupby(["month", "status"])[TARGET_COLUMN]
        .agg(count="size", default_rate="mean")
        .reset_index()
    )
    summary["share"] = summary["count"] / len(frame)
    return summary[["month", "status", "count", "share", "default_rate"]]


def _standardized_mean_difference(reference: pd.Series, comparison: pd.Series) -> float:
    pooled_standard_deviation = np.sqrt((reference.var() + comparison.var()) / 2)
    if pooled_standard_deviation == 0:
        return 0.0 if reference.mean() == comparison.mean() else float("inf")
    return float((comparison.mean() - reference.mean()) / pooled_standard_deviation)


def _max_category_proportion_difference(
    reference: pd.Series, comparison: pd.Series
) -> float:
    reference_share = reference.value_counts(normalize=True)
    comparison_share = comparison.value_counts(normalize=True)
    categories = reference_share.index.union(comparison_share.index)
    differences = reference_share.reindex(categories, fill_value=0) - comparison_share.reindex(
        categories, fill_value=0
    )
    return float(differences.abs().max())


def compare_split_distributions(splits: DataSplits) -> dict[str, object]:
    """Compare validation and test distributions with the training reference."""
    numeric_rows = []
    for feature in NUMERIC_COLUMNS:
        numeric_rows.append(
            {
                "feature": feature,
                "validation_standardized_mean_difference": _standardized_mean_difference(
                    splits.X_train[feature], splits.X_validation[feature]
                ),
                "test_standardized_mean_difference": _standardized_mean_difference(
                    splits.X_train[feature], splits.X_test[feature]
                ),
            }
        )
    categorical_rows = []
    for feature in CATEGORICAL_COLUMNS:
        categorical_rows.append(
            {
                "feature": feature,
                "validation_max_proportion_difference": _max_category_proportion_difference(
                    splits.X_train[feature], splits.X_validation[feature]
                ),
                "test_max_proportion_difference": _max_category_proportion_difference(
                    splits.X_train[feature], splits.X_test[feature]
                ),
            }
        )
    return {
        "target_rates": {
            "train": float(splits.y_train.mean()),
            "validation": float(splits.y_validation.mean()),
            "test": float(splits.y_test.mean()),
        },
        "numeric": pd.DataFrame(numeric_rows).set_index("feature").loc[list(NUMERIC_COLUMNS)],
        "categorical": pd.DataFrame(categorical_rows)
        .set_index("feature")
        .loc[list(CATEGORICAL_COLUMNS)],
    }


def _configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_eda_figures(
    frame: pd.DataFrame, splits: DataSplits, output_directory: str | Path
) -> list[Path]:
    """Generate the fixed set of decision-relevant EDA figures."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _configure_plotting()
    paths: list[Path] = []

    target_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(["0: Non-default / 未违约", "1: Default / 违约"], target_counts.values)
    axis.set_title("Target distribution / 目标分布")
    axis.set_ylabel("Observations / 样本数")
    axis.set_ylim(0, target_counts.max() * 1.18)
    for position, count in enumerate(target_counts.values):
        axis.text(
            position,
            count + target_counts.max() * 0.015,
            f"{count:,}\n{count / len(frame):.2%}",
            ha="center",
            va="bottom",
        )
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[0]))

    representative_numeric = (
        "credit_limit",
        "age",
        "bill_amount_sep",
        "payment_amount_sep",
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, feature in zip(axes.flat, representative_numeric, strict=True):
        display_values = frame[feature]
        title_suffix = ""
        if feature.startswith(("bill_amount_", "payment_amount_")):
            lower, upper = display_values.quantile([0.01, 0.99])
            display_values = display_values[display_values.between(lower, upper)]
            title_suffix = " (central 98% / 中间 98%)"
        sns.histplot(display_values, bins=35, ax=axis, color="#2878B5")
        axis.set_title(f"{feature} distribution / 分布{title_suffix}")
        axis.set_xlabel(feature)
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[1]))

    categorical_summary = summarize_categorical_features(frame)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, feature in zip(axes, ("sex", "education", "marital_status"), strict=True):
        subset = categorical_summary[categorical_summary["feature"] == feature]
        sns.barplot(data=subset, x="value", y="default_rate", ax=axis, color="#3C9D76")
        axis.set_title(f"{feature}: default rate / 违约率")
        axis.set_ylabel("Default rate / 违约率")
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[2]))

    repayment_summary = summarize_repayment_status(frame)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.lineplot(
        data=repayment_summary,
        x="status",
        y="share",
        hue="month",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Status frequency share (log scale) / 状态占比（对数轴）")
    axes[0].set_ylabel("Share / 占比")
    sns.lineplot(
        data=repayment_summary,
        x="status",
        y="default_rate",
        hue="month",
        marker="o",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Status and default rate / 状态与违约率")
    axes[1].set_ylabel("Default rate / 违约率")
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[3]))

    correlations = frame.loc[:, list(NUMERIC_COLUMNS) + [TARGET_COLUMN]].corr()
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
    axis.set_yticks(range(len(correlations.index)), correlations.index)
    figure.colorbar(image, ax=axis, label="Correlation / 相关系数")
    axis.set_title("Numeric correlations (non-causal) / 数值相关性（非因果）")
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[4]))

    comparison = compare_split_distributions(splits)
    numeric = comparison["numeric"].abs().max()
    categorical = comparison["categorical"].abs().max()
    rates = comparison["target_rates"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(list(rates), list(rates.values()))
    axes[0].set_title("Default rate by split / 各切分违约率")
    axes[0].set_ylabel("Default rate / 违约率")
    axes[1].bar(
        ["Validation\nnumeric SMD", "Test\nnumeric SMD", "Validation\ncategory diff", "Test\ncategory diff"],
        [numeric.iloc[0], numeric.iloc[1], categorical.iloc[0], categorical.iloc[1]],
    )
    axes[1].set_title("Maximum split differences / 最大切分差异")
    paths.append(_save_figure(figure, output_directory / FIGURE_NAMES[5]))
    return paths


def _markdown_rows(rows: list[list[str]], headers: list[str]) -> str:
    separator = ["---"] * len(headers)
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [headers, separator, *rows]
    )


def _numeric_table(summary: pd.DataFrame, *, english: bool) -> str:
    headers = (
        ["Feature", "Minimum", "Median", "P99", "Maximum", "Zero share", "Negative share", "Skewness"]
        if english
        else ["变量", "最小值", "中位数", "P99", "最大值", "零值比例", "负值比例", "偏度"]
    )
    rows = [
        [
            feature,
            f"{row.minimum:,.2f}",
            f"{row['median']:,.2f}",
            f"{row.p99:,.2f}",
            f"{row.maximum:,.2f}",
            f"{row.zero_share:.2%}",
            f"{row.negative_share:.2%}",
            f"{row.skewness:.2f}",
        ]
        for feature, row in summary.iterrows()
    ]
    return _markdown_rows(rows, headers)


def _category_table(summary: pd.DataFrame, *, english: bool) -> str:
    selected = summary[summary["feature"].isin(("sex", "education", "marital_status", "repayment_status_sep"))]
    headers = (
        ["Feature", "Code", "Count", "Share", "Default rate"]
        if english
        else ["变量", "编码", "数量", "占比", "违约率"]
    )
    rows = [
        [
            str(row.feature),
            str(row.value),
            f"{int(row['count']):,}",
            f"{row.share:.2%}",
            f"{row.default_rate:.2%}",
        ]
        for _, row in selected.iterrows()
    ]
    return _markdown_rows(rows, headers)


def _split_table(comparison: dict[str, object], *, english: bool) -> str:
    rates = comparison["target_rates"]
    headers = ["Split", "Default rate"] if english else ["切分", "违约率"]
    labels = {"train": "训练集", "validation": "验证集", "test": "测试集"}
    rows = [
        [split if english else labels[split], f"{rate:.2%}"]
        for split, rate in rates.items()
    ]
    return _markdown_rows(rows, headers)


def _repayment_month_table(summary: pd.DataFrame, *, english: bool) -> str:
    headers = (
        ["Month field", "Observed codes", "Most common code", "Most common share", "Smallest group"]
        if english
        else ["月份字段", "观察编码数", "最常见编码", "最常见占比", "最小组样本数"]
    )
    rows = []
    for month, month_summary in summary.groupby("month", sort=False):
        most_common = month_summary.loc[month_summary["count"].idxmax()]
        rows.append(
            [
                month,
                str(len(month_summary)),
                str(most_common["status"]),
                f"{most_common['share']:.2%}",
                f"{int(month_summary['count'].min()):,}",
            ]
        )
    return _markdown_rows(rows, headers)


def format_eda_report(frame: pd.DataFrame, splits: DataSplits) -> str:
    """Render observed EDA findings as a complete bilingual Markdown report."""
    target_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    target_rate = float(frame[TARGET_COLUMN].mean())
    numeric = summarize_numeric_features(frame)
    categorical = summarize_categorical_features(frame)
    repayment = summarize_repayment_status(frame)
    comparison = compare_split_distributions(splits)
    numeric_difference = comparison["numeric"].abs().max()
    categorical_difference = comparison["categorical"].abs().max()
    target_correlations = (
        frame.loc[:, list(NUMERIC_COLUMNS) + [TARGET_COLUMN]]
        .corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .abs()
        .sort_values(ascending=False)
    )
    top_correlations = ", ".join(
        f"`{feature}` ({value:.3f})" for feature, value in target_correlations.head(5).items()
    )
    return f"""# 探索性数据分析报告（中文）

本报告由 `credit_risk.analysis` 基于经过校验的 UCI 原始数据实际运行生成。分析对象是已有客户行为风险；数据不支持真正的 OOT 验证。

## 目标分布

- 样本总数：{len(frame):,}。
- 未违约（`0`）：{target_counts.get(0, 0):,}（{target_counts.get(0, 0) / len(frame):.2%}）。
- 违约（`1`）：{target_counts.get(1, 0):,}（{target_rate:.2%}）。

类别存在中等不平衡，因此 Accuracy 不作为主要模型指标。后续基线使用 ROC-AUC、PR-AUC、KS、Brier Score 和校准结果。

![目标分布](figures/eda_target_distribution.png)

## 数值变量

账单金额允许出现负值，付款金额存在大量零值；数据来源没有证据表明它们是录入错误，因此本阶段保留原值。金额变量明显右偏，逻辑回归 Pipeline 会在训练集内标准化，但本基线不删除、截尾或对数变换。为避免极端值压缩主体形状，金额直方图只展示中心 98% 的观察值；下表仍报告完整范围，模型仍使用全部原值。

{_numeric_table(numeric, english=False)}

![代表性数值分布](figures/eda_numeric_distributions.png)

## 类别和还款状态

下表列出人口属性和最近一个月还款状态的实际频数与组内违约率。来源未定义的编码继续作为独立类别保留，不能擅自赋予业务含义。六个月还款状态图显示的是关联，不是因果关系。高还款状态编码中的小样本组会产生波动较大的违约率，不应仅根据单个点下结论。

{_category_table(categorical, english=False)}

六个月还款状态频数摘要如下。完整频数曲线使用对数纵轴，以便同时看见常见编码和极小尾部组。

{_repayment_month_table(repayment, english=False)}

![类别违约率](figures/eda_category_default_rates.png)

![还款状态与违约率](figures/eda_repayment_status.png)

## 相关性

按与目标绝对相关系数排序的前五个数值变量为：{top_correlations}。这些相关性用于识别候选关系和共线性，不是因果证据。

![数值相关矩阵](figures/eda_correlation_heatmap.png)

## 切分分布检查

{_split_table(comparison, english=False)}

- 验证集最大绝对数值 SMD：{numeric_difference['validation_standardized_mean_difference']:.4f}；测试集：{numeric_difference['test_standardized_mean_difference']:.4f}。
- 验证集最大类别比例差：{categorical_difference['validation_max_proportion_difference']:.4f}；测试集：{categorical_difference['test_max_proportion_difference']:.4f}。

这些结果仅检查分层随机切分的相似程度。观察到的差异不能称为时间漂移，也不能把该测试集称为 OOT。

![切分分布检查](figures/eda_split_comparison.png)

## 限制

历史账单、付款和还款状态通常无法在新客户首次申请时取得，因此结果不能解释为新客户贷前审批。相关性和组间违约率差异不证明因果效应；公开教学数据也不能代表当前金融机构的真实客户组合或经济周期。

---

# Exploratory Data Analysis Report (English)

This report was generated by executing `credit_risk.analysis` against the verified UCI source data. It concerns behavioral risk for existing customers; the data do not support genuine OOT validation.

## Target distribution

- Total observations: {len(frame):,}.
- Non-default (`0`): {target_counts.get(0, 0):,} ({target_counts.get(0, 0) / len(frame):.2%}).
- Default (`1`): {target_counts.get(1, 0):,} ({target_rate:.2%}).

The classes are moderately imbalanced, so Accuracy is not a primary model metric. The baseline uses ROC-AUC, PR-AUC, KS, Brier Score, and calibration results.

![Target distribution](figures/eda_target_distribution.png)

## Numeric features

Bill amounts can be negative and payment amounts contain many zeros. The source provides no evidence that these values are input errors, so this stage retains them. Amount features are strongly right-skewed. The logistic-regression pipeline standardizes them using training data, but this baseline does not delete, winsorize, or log-transform them. To keep extreme values from compressing the main shape, amount histograms display the central 98% of observations; the table still reports full ranges, and the model still uses every original value.

{_numeric_table(numeric, english=True)}

![Representative numeric distributions](figures/eda_numeric_distributions.png)

## Categories and repayment status

The table reports observed frequencies and within-group default rates for demographics and the most recent repayment status. Source-undefined codes remain separate categories and receive no invented business meaning. The six-month repayment-status figure shows associations that are not causal. Rates for small groups among high repayment-status codes are volatile and should not support conclusions from individual points.

{_category_table(categorical, english=True)}

The six-month repayment-status frequency summary follows. The full frequency panel uses a logarithmic vertical axis so both common codes and very small tail groups remain visible.

{_repayment_month_table(repayment, english=True)}

![Category default rates](figures/eda_category_default_rates.png)

![Repayment status and default rate](figures/eda_repayment_status.png)

## Correlation

The five numeric features with the highest absolute target correlations are: {top_correlations}. These correlations help identify candidate relationships and collinearity; they are not causal evidence.

![Numeric correlation matrix](figures/eda_correlation_heatmap.png)

## Split-distribution checks

{_split_table(comparison, english=True)}

- Maximum absolute numeric SMD: {numeric_difference['validation_standardized_mean_difference']:.4f} for validation and {numeric_difference['test_standardized_mean_difference']:.4f} for test.
- Maximum category-proportion difference: {categorical_difference['validation_max_proportion_difference']:.4f} for validation and {categorical_difference['test_max_proportion_difference']:.4f} for test.

These results only check similarity after a stratified random split. Observed differences are not temporal drift, and the test set is not OOT.

![Split-distribution checks](figures/eda_split_comparison.png)

## Limitations

Historical bills, payments, and repayment status are generally unavailable at a new customer's first application, so results do not represent new-customer underwriting. Correlations and group default-rate differences do not prove causal effects. This public teaching dataset also does not represent a current financial institution's actual customer mix or economic cycle.
"""


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("data/raw") / RAW_XLS_NAME)
    parser.add_argument("--output", type=Path, default=Path("reports/eda_report.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    frame = load_dataset(args.path)
    splits = split_dataset(frame)
    generate_eda_figures(frame, splits, args.figures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_eda_report(frame, splits), encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    _main()
