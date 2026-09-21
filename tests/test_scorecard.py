import hashlib
import inspect
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.config import CVConfig, load_config
from credit_risk.modeling import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from credit_risk.scorecard import (
    FittedScorecard,
    ScoreMapping,
    build_scorecard_pipeline,
    fit_scorecard_development,
    format_scorecard_report,
    load_scorecard_development_data,
    select_scorecard_c,
)
from credit_risk.woe import WOETransformer


def fit_small_scorecard() -> tuple[FittedScorecard, pd.DataFrame]:
    X = pd.DataFrame(
        {
            "category": ["a", "a", "b", "b", "a", "b", None, "rare"] * 2,
            "value": np.arange(16, dtype=float),
        }
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 2)
    pipeline = Pipeline(
        [
            (
                "woe",
                WOETransformer(
                    categorical_features=("category",),
                    numeric_features=("value",),
                    max_bins=3,
                    min_bin_fraction=0.15,
                ),
            ),
            ("classifier", LogisticRegression(C=0.1, random_state=42)),
        ]
    ).fit(X, y)
    return FittedScorecard(pipeline, ScoreMapping(), 0.1), X


def make_project_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(71)
    target = np.tile([0, 0, 0, 1], rows // 4)
    data: dict[str, np.ndarray] = {
        "customer_id": np.arange(1, rows + 1),
        "default_next_month": target,
    }
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.integers(0, 3, size=rows)
    for column in NUMERIC_COLUMNS:
        data[column] = rng.normal(100, 15, size=rows)
    data["repayment_status_sep"] = target + rng.integers(0, 2, size=rows)
    return pd.DataFrame(data)


def reduced_config():
    config = load_config("configs/experiment_v2.json")
    search = replace(config.model_search, scorecard_c_values=(0.01, 0.1))
    return replace(config, cv=CVConfig(3, True), model_search=search)


def test_score_mapping_base_odds_pdo_and_probability_round_trip() -> None:
    mapping = ScoreMapping(base_score=600, base_odds=20, pdo=50)
    pd_at_20_to_1 = 1 / 21
    pd_at_40_to_1 = 1 / 41

    assert mapping.probability_to_score(pd_at_20_to_1) == pytest.approx(600)
    assert mapping.probability_to_score(pd_at_40_to_1) == pytest.approx(650)
    probabilities = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
    scores = mapping.probability_to_score(probabilities)
    assert mapping.score_to_probability(scores) == pytest.approx(probabilities)
    assert np.all(np.diff(scores) < 0)


def test_total_score_equals_base_plus_contributions_and_inverts_raw_pd() -> None:
    scorecard, X = fit_small_scorecard()
    sample = X.iloc[[0, 6]].copy()
    sample.loc[sample.index[0], "category"] = "unseen"
    components = scorecard.score_components(sample)
    contribution_columns = [
        column
        for column in components
        if column.endswith("_points") and column != "base_points"
    ]

    reconstructed = components["base_points"] + components[contribution_columns].sum(axis=1)
    np.testing.assert_allclose(components["total_score"], reconstructed)
    np.testing.assert_allclose(components["total_score"], scorecard.score(sample))
    np.testing.assert_allclose(
        scorecard.mapping.score_to_probability(components["total_score"]),
        components["raw_pd"],
    )
    assert np.isfinite(components.to_numpy()).all()


def test_points_table_and_reason_codes_are_auditable() -> None:
    scorecard, X = fit_small_scorecard()
    table = scorecard.points_table()
    reasons = scorecard.reason_codes(X.iloc[:2], top_n=2)

    assert {"coefficient", "points", "woe", "feature"} <= set(table.columns)
    np.testing.assert_allclose(
        table["points"],
        -scorecard.mapping.factor * table["coefficient"] * table["woe"],
    )
    assert len(reasons) == 2
    assert all(len(row) <= 2 for row in reasons)
    assert all(item["deduction_points"] < 0 for row in reasons for item in row)


def test_scorecard_pipeline_is_unweighted_and_fold_safe() -> None:
    config = reduced_config()
    pipeline = build_scorecard_pipeline(config, c_value=0.1)

    assert tuple(pipeline.named_steps) == ("woe", "classifier")
    assert pipeline.named_steps["classifier"].class_weight is None
    assert pipeline.named_steps["classifier"].C == 0.1
    assert not hasattr(pipeline.named_steps["woe"], "binning_table_")


def test_training_only_cv_selection_is_reproducible() -> None:
    frame = make_project_frame()
    config = reduced_config()
    X = frame.loc[:, config.feature_columns]
    y = frame[config.target_column]

    first = select_scorecard_c(X, y, config)
    second = select_scorecard_c(X, y, config)

    assert first.selected_c == second.selected_c
    assert len(first.fold_results) == 6
    assert set(first.fold_results["fit_rows"]) == {80}
    assert set(first.fold_results["score_rows"]) == {40}
    pd.testing.assert_frame_equal(
        first.summary.drop(columns="total_seconds"),
        second.summary.drop(columns="total_seconds"),
    )
    assert list(inspect.signature(select_scorecard_c).parameters) == [
        "X_train",
        "y_train",
        "config",
    ]


def test_fixed_loader_ignores_test_labels(tmp_path: Path) -> None:
    frame = make_project_frame()
    assignments = pd.DataFrame(
        {
            "customer_id": frame["customer_id"],
            "split": ["train"] * 72 + ["validation"] * 24 + ["test"] * 24,
        }
    )
    path = tmp_path / "assignments.csv"
    assignments.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = load_config("configs/experiment_v2.json")
    first = load_scorecard_development_data(
        frame, path, config=config, expected_assignment_sha256=digest
    )
    altered = frame.copy()
    altered.loc[96:, "default_next_month"] = 1 - altered.loc[96:, "default_next_month"]
    second = load_scorecard_development_data(
        altered, path, config=config, expected_assignment_sha256=digest
    )

    pd.testing.assert_series_equal(first.y_train, second.y_train)
    pd.testing.assert_series_equal(first.y_validation, second.y_validation)
    assert not hasattr(first, "y_test")
    assert not hasattr(first, "X_test")


def test_development_report_is_bilingual_and_contains_actual_metrics(tmp_path: Path) -> None:
    frame = make_project_frame()
    assignments = pd.DataFrame(
        {
            "customer_id": frame["customer_id"],
            "split": ["train"] * 72 + ["validation"] * 24 + ["test"] * 24,
        }
    )
    path = tmp_path / "assignments.csv"
    assignments.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = reduced_config()
    data = load_scorecard_development_data(
        frame, path, config=config, expected_assignment_sha256=digest
    )
    result = fit_scorecard_development(data, config)
    report = format_scorecard_report(
        result,
        data,
        run_id="test_scorecard",
        data_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    observed_auc = result.validation_metrics["woe_scorecard"]["roc_auc"]

    assert report.startswith("# 评分卡开发报告（中文）")
    assert report.index("# Scorecard Development Report (English)") > 0
    assert report.count(f"{observed_auc:.6f}") == 2
    assert report.count("test_scorecard") == 2
    assert "不是行业统一标准" in report
    assert "not universal industry standards" in report
    assert "校准后 PD" in report
    assert "calibrated decision PD" in report
