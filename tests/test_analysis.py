from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from credit_risk.analysis import (
    compare_split_distributions,
    format_eda_report,
    generate_eda_figures,
    summarize_categorical_features,
    summarize_numeric_features,
    summarize_repayment_status,
)
from credit_risk.modeling import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, split_dataset


def make_analysis_frame(rows: int = 40) -> pd.DataFrame:
    target = np.tile([0, 0, 0, 1], rows // 4)
    data: dict[str, np.ndarray] = {
        "customer_id": np.arange(1, rows + 1),
        "default_next_month": target,
    }
    for column in CATEGORICAL_COLUMNS:
        data[column] = np.tile([0, 1, 2, 1], rows // 4)
    for offset, column in enumerate(NUMERIC_COLUMNS):
        data[column] = np.arange(rows, dtype=float) + offset + 1
    data["bill_amount_sep"][:10] *= -1
    data["payment_amount_sep"][:20] = 0
    return pd.DataFrame(data)


def test_numeric_summary_reports_distribution_diagnostics() -> None:
    summary = summarize_numeric_features(make_analysis_frame())

    assert list(summary.index) == list(NUMERIC_COLUMNS)
    assert summary.loc["bill_amount_sep", "negative_share"] == 0.25
    assert summary.loc["payment_amount_sep", "zero_share"] == 0.5
    assert {
        "minimum",
        "p01",
        "median",
        "p99",
        "maximum",
        "zero_share",
        "negative_share",
        "skewness",
    } <= set(summary.columns)


def test_categorical_summary_reports_counts_shares_and_default_rates() -> None:
    summary = summarize_categorical_features(make_analysis_frame())
    sex_one = summary[(summary["feature"] == "sex") & (summary["value"] == 1)].iloc[0]

    assert sex_one["count"] == 20
    assert sex_one["share"] == 0.5
    assert sex_one["default_rate"] == 0.5
    assert set(summary["feature"]) == set(CATEGORICAL_COLUMNS)


def test_repayment_summary_covers_frequency_and_default_rate_for_all_months() -> None:
    frame = make_analysis_frame()
    summary = summarize_repayment_status(frame)
    repayment_columns = {
        column for column in CATEGORICAL_COLUMNS if column.startswith("repayment_status_")
    }

    assert set(summary.columns) == {"month", "status", "count", "share", "default_rate"}
    assert set(summary["month"]) == repayment_columns
    assert (summary.groupby("month")["count"].sum() == len(frame)).all()
    assert np.allclose(summary.groupby("month")["share"].sum(), 1.0)


def test_split_comparison_reports_target_numeric_and_categorical_differences() -> None:
    splits = split_dataset(make_analysis_frame())
    comparison = compare_split_distributions(splits)

    assert set(comparison) == {"target_rates", "numeric", "categorical"}
    assert set(comparison["target_rates"]) == {"train", "validation", "test"}
    assert list(comparison["numeric"].index) == list(NUMERIC_COLUMNS)
    assert list(comparison["categorical"].index) == list(CATEGORICAL_COLUMNS)
    assert {
        "validation_standardized_mean_difference",
        "test_standardized_mean_difference",
    } <= set(comparison["numeric"].columns)
    assert {
        "validation_max_proportion_difference",
        "test_max_proportion_difference",
    } <= set(comparison["categorical"].columns)


def test_eda_report_is_complete_bilingual_and_reuses_observed_numbers() -> None:
    frame = make_analysis_frame()
    report = format_eda_report(frame, split_dataset(frame))

    assert report.startswith("# 探索性数据分析报告（中文）")
    assert report.index("# Exploratory Data Analysis Report (English)") > report.index(
        "# 探索性数据分析报告（中文）"
    )
    assert report.count("25.00%") >= 2
    assert "不是因果" in report
    assert "not causal" in report
    assert "小样本" in report
    assert "small groups" in report
    assert "OOT" in report
    assert "最小组样本数" in report
    assert "Smallest group" in report
    assert all(
        column in report
        for column in CATEGORICAL_COLUMNS
        if column.startswith("repayment_status_")
    )


def test_generate_eda_figures_creates_the_decision_relevant_set(tmp_path: Path) -> None:
    frame = make_analysis_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PendingDeprecationWarning)
        paths = generate_eda_figures(frame, split_dataset(frame), tmp_path)

    assert {path.name for path in paths} == {
        "eda_target_distribution.png",
        "eda_numeric_distributions.png",
        "eda_category_default_rates.png",
        "eda_repayment_status.png",
        "eda_correlation_heatmap.png",
        "eda_split_comparison.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)
