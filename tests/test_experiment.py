from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from credit_risk.experiment import (
    format_model_report,
    generate_model_figures,
    run_baseline_experiment,
)
from credit_risk.modeling import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def make_experiment_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    target = np.tile([0, 0, 0, 1], rows // 4)
    data: dict[str, np.ndarray] = {
        "customer_id": np.arange(1, rows + 1),
        "default_next_month": target,
    }
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.integers(0, 3, size=rows)
    for offset, column in enumerate(NUMERIC_COLUMNS):
        data[column] = rng.normal(100 + offset, 20, size=rows)
    data["repayment_status_sep"] = target + rng.integers(0, 2, size=rows)
    return pd.DataFrame(data)


def test_experiment_uses_one_split_and_training_only_preprocessor_fits() -> None:
    result = run_baseline_experiment(make_experiment_frame())

    assert set(result.models) == {"dummy", "logistic", "logistic_balanced"}
    assert set(result.metrics) == {"validation", "test"}
    assert set(result.metrics["validation"]) == set(result.models)
    assert set(result.metrics["test"]) == set(result.models)
    assert (len(result.splits.X_train), len(result.splits.X_validation), len(result.splits.X_test)) == (
        72,
        24,
        24,
    )
    for model in result.models.values():
        scaler = model.named_steps["preprocessor"].named_transformers_["numeric"]
        assert scaler.n_samples_seen_ == len(result.splits.X_train)
    assert all(len(values) == 24 for values in result.validation_probabilities.values())
    assert all(len(values) == 24 for values in result.test_probabilities.values())


def test_model_report_is_bilingual_and_contains_observed_metrics() -> None:
    result = run_baseline_experiment(make_experiment_frame())
    report = format_model_report(result)
    observed_auc = result.metrics["test"]["logistic"]["roc_auc"]

    assert report.startswith("# 逻辑回归基线模型报告（中文）")
    assert report.index("# Logistic Regression Baseline Model Report (English)") > 0
    assert f"{observed_auc:.4f}" in report
    assert report.count("0.5") >= 2
    assert report.count("max_iter=2000") == 2
    assert "10 个等宽" in report
    assert "10 equal-width" in report
    assert "校准箱" in report
    assert "calibration bins" in report
    assert "不是 OOT" in report
    assert "not OOT" in report


def test_generate_model_figures_creates_fixed_evaluation_set(tmp_path: Path) -> None:
    result = run_baseline_experiment(make_experiment_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("error", PendingDeprecationWarning)
        paths = generate_model_figures(result, tmp_path)

    assert {path.name for path in paths} == {
        "model_roc_curves.png",
        "model_pr_curves.png",
        "model_calibration.png",
        "model_confusion_matrices.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)
