import copy
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_validate

from credit_risk.config import ConfigError, load_config, validate_config
from credit_risk.experiment import run_development_experiment
from credit_risk.modeling import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_model_pipelines,
)
from credit_risk.protocol import (
    DevelopmentData,
    build_stratified_cv,
    create_protocol_split,
    duplicate_split_summary,
    format_protocol_report,
    protocol_manifest,
)


CONFIG_PATH = Path("configs/experiment_v2.json")


def make_frame(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    target = np.tile([0, 0, 0, 1], rows // 4 + 1)[:rows]
    data: dict[str, np.ndarray] = {
        "customer_id": np.arange(10_001, 10_001 + rows),
        "default_next_month": target,
    }
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.integers(0, 3, size=rows)
    for column in NUMERIC_COLUMNS:
        data[column] = rng.normal(100, 10, size=rows)
    return pd.DataFrame(data)


def raw_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_repository_config_is_valid_and_complete() -> None:
    config = load_config(CONFIG_PATH)

    assert config.random_seed == 42
    assert config.split.train == pytest.approx(0.6)
    assert config.split.validation == pytest.approx(0.2)
    assert config.split.test == pytest.approx(0.2)
    assert config.cv.folds == 5
    assert config.woe.max_bins == 5
    assert config.woe.min_bin_fraction == pytest.approx(0.05)
    assert config.woe.smoothing == pytest.approx(0.5)
    assert config.feature_columns == FEATURE_COLUMNS
    assert len(config.sha256) == 64


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("random_seed",), -1, "random_seed"),
        (("split", "test"), 0, "split.test"),
        (("split", "train"), 0.5, "sum to 1"),
        (("split", "stratify"), False, "stratify"),
        (("cross_validation", "folds"), 1, "folds"),
        (("woe", "max_bins"), 0, "woe.max_bins"),
        (("woe", "min_bin_fraction"), 1.0, "must be below 1"),
        (("woe", "smoothing"), 0, "woe.smoothing"),
        (("data", "sha256"), "bad", "64-character"),
        (("model_search", "max_candidates"), 13, "must not exceed 12"),
        (("score_mapping", "pdo"), None, "must be numeric"),
        (("calibration", "folds"), 1, "calibration.folds"),
        (("strategy", "approve_when"), "pd <= threshold", "approve_when"),
        (("duplicates", "main_protocol"), "drop", "must be retain"),
    ],
)
def test_invalid_configuration_is_rejected(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = copy.deepcopy(raw_config())
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index,assignment]
    cursor[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_protocol_split_is_reproducible_mutually_exclusive_and_complete() -> None:
    frame = make_frame()
    config = load_config(CONFIG_PATH)

    first = create_protocol_split(frame, config)
    second = create_protocol_split(frame, config)

    assert first.assignment_sha256 == second.assignment_sha256
    assert first.assignments.equals(second.assignments)
    assert first.assignments["customer_id"].is_unique
    assert set(first.assignments["customer_id"]) == set(frame["customer_id"])
    assert first.assignments["split"].value_counts().to_dict() == {
        "train": 120,
        "validation": 40,
        "test": 40,
    }
    id_sets = {
        name: set(first.assignments.loc[first.assignments["split"] == name, "customer_id"])
        for name in ("train", "validation", "test")
    }
    assert id_sets["train"].isdisjoint(id_sets["validation"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["validation"].isdisjoint(id_sets["test"])


def test_development_entry_has_no_test_data_and_does_not_depend_on_test_labels() -> None:
    frame = make_frame(120)
    split = create_protocol_split(frame, load_config(CONFIG_PATH))
    assert {field.name for field in fields(DevelopmentData)} == {
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
    }

    first = run_development_experiment(split.development)
    altered_test_labels = 1 - split.final_evaluation.y_test
    second = run_development_experiment(split.development)

    assert not altered_test_labels.equals(split.final_evaluation.y_test)
    assert first.validation_metrics == second.validation_metrics
    assert set(first.validation_probabilities) == set(second.validation_probabilities)


def test_preprocessing_is_fitted_inside_each_cross_validation_fold() -> None:
    frame = make_frame(200)
    split = create_protocol_split(frame, load_config(CONFIG_PATH))
    model = build_model_pipelines()["logistic"]

    result = cross_validate(
        model,
        split.development.X_train,
        split.development.y_train,
        cv=build_stratified_cv(load_config(CONFIG_PATH)),
        scoring="roc_auc",
        return_estimator=True,
    )

    expected_fold_training_rows = 96
    assert len(result["estimator"]) == 5
    for estimator in result["estimator"]:
        scaler = estimator.named_steps["preprocessor"].named_transformers_["numeric"]
        assert scaler.n_samples_seen_ == expected_fold_training_rows


def test_duplicate_summary_reports_cross_split_groups() -> None:
    frame = make_frame(40)
    assignments = pd.DataFrame(
        {"customer_id": frame["customer_id"], "split": ["train"] * 20 + ["test"] * 20}
    )
    copied_columns = [column for column in frame.columns if column != "customer_id"]
    frame.loc[39, copied_columns] = frame.loc[0, copied_columns].to_numpy()

    summary = duplicate_split_summary(frame, assignments, id_column="customer_id")

    assert summary["duplicate_rows_after_first"] == 1
    assert summary["duplicate_groups"] == 1
    assert summary["cross_split_groups"] == 1
    assert summary["rows_in_cross_split_groups"] == 2


def test_protocol_report_is_bilingual_and_matches_manifest() -> None:
    frame = make_frame(200)
    config = load_config(CONFIG_PATH)
    split = create_protocol_split(frame, config)
    duplicates = duplicate_split_summary(frame, split.assignments, id_column="customer_id")
    manifest = protocol_manifest(
        config,
        split,
        duplicates,
        data_sha256="a" * 64,
        run_id="test_run",
    )

    report = format_protocol_report(manifest)

    assert report.startswith("# 实验协议报告（中文）")
    assert report.index("# Experiment Protocol Report (English)") > 0
    assert report.count(split.assignment_sha256) == 2
    assert report.count("30.0000%") == 0
    assert report.count("25.0000%") == 6
    assert "不是梯形积分 PR-AUC" in report
    assert "not trapezoidal PR-AUC" in report
