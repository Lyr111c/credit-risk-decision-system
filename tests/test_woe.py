import inspect
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.config import load_config
from credit_risk.modeling import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from credit_risk.woe import (
    WOETransformer,
    format_woe_report,
    load_woe_development_data,
    woe_summary,
)


def test_hand_calculated_smoothed_woe_iv_with_all_good_and_all_bad_bins() -> None:
    X = pd.DataFrame({"value": [0.0, 0.0, 1.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])
    transformer = WOETransformer(
        numeric_features=("value",),
        max_bins=2,
        min_bin_fraction=0.25,
        smoothing=0.5,
    ).fit(X, y)

    table = transformer.binning_table_.sort_values("bin_order")
    assert table["good_count"].tolist() == [2, 0]
    assert table["bad_count"].tolist() == [0, 2]
    assert table["woe"].tolist() == pytest.approx([math.log(5), -math.log(5)])
    assert transformer.iv_["value"] == pytest.approx(4 / 3 * math.log(5))
    assert table["iv_component"].sum() == pytest.approx(transformer.iv_["value"])
    assert np.isfinite(table[["woe", "iv_component", "total_iv"]]).all().all()


def test_numeric_intervals_are_right_closed_and_cover_extreme_values() -> None:
    X = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]})
    transformer = WOETransformer(
        numeric_features=("value",), max_bins=2, min_bin_fraction=0.25
    ).fit(X, [0, 0, 1, 1])
    first_woe, second_woe = transformer.binning_table_["woe"].tolist()

    transformed = transformer.transform(
        pd.DataFrame({"value": [-1e20, 1.5, 1.5000001, 1e20]})
    )["value_woe"]

    assert transformed.tolist() == pytest.approx(
        [first_woe, first_woe, second_woe, second_woe]
    )
    table = transformer.binning_table_
    assert table.iloc[0]["lower_bound"] == -np.inf
    assert table.iloc[-1]["upper_bound"] == np.inf
    assert table["right_closed"].tolist() == [True, True]


def test_missing_other_and_unseen_category_rules() -> None:
    X = pd.DataFrame({"category": ["a", "a", "a", "b", None, None]})
    transformer = WOETransformer(
        categorical_features=("category",), min_bin_fraction=0.3
    ).fit(X, [0, 0, 1, 1, 0, 1])
    table = transformer.binning_table_.set_index("bin_id")

    transformed = transformer.transform(
        pd.DataFrame({"category": ["unseen", "b", None, "a"]})
    )["category_woe"]

    assert transformed.iloc[0] == pytest.approx(table.loc["other", "woe"])
    assert transformed.iloc[1] == pytest.approx(table.loc["other", "woe"])
    assert transformed.iloc[2] == pytest.approx(table.loc["missing", "woe"])
    assert transformed.iloc[3] == pytest.approx(table.loc["category_0", "woe"])
    assert table.loc["other", "categories"] == '["b"]'


def test_unseen_category_is_neutral_when_training_has_no_other_bin() -> None:
    X = pd.DataFrame({"category": ["a", "a", "b", "b"]})
    transformer = WOETransformer(
        categorical_features=("category",), min_bin_fraction=0.25
    ).fit(X, [0, 1, 0, 1])

    transformed = transformer.transform(pd.DataFrame({"category": ["new", None]}))

    assert transformed["category_woe"].tolist() == [0.0, 0.0]


def test_constant_repeated_quantiles_and_all_missing_are_stable() -> None:
    X = pd.DataFrame(
        {
            "constant": [7.0] * 8,
            "repeated": [0.0] * 6 + [1.0, 2.0],
            "all_missing": [np.nan] * 8,
        }
    )
    transformer = WOETransformer(
        numeric_features=("constant", "repeated", "all_missing"),
        max_bins=5,
        min_bin_fraction=0.25,
    ).fit(X, [0, 1] * 4)

    summary = woe_summary(transformer).set_index("feature")
    assert summary.loc["constant", "bin_count"] == 1
    assert summary.loc["repeated", "bin_count"] <= 4
    assert (
        transformer.binning_table_
        .query("feature == 'repeated' and bin_id != 'missing'")["sample_count"]
        .min()
        >= 2
    )
    assert transformer.binning_table_.query("feature == 'all_missing'")[
        "bin_id"
    ].tolist() == ["missing"]
    transformed = transformer.transform(
        pd.DataFrame(
            {"constant": [7.0], "repeated": [100.0], "all_missing": [1.0]}
        )
    )
    assert np.isfinite(transformed.to_numpy()).all()
    assert transformed.loc[0, "all_missing_woe"] == 0.0


@pytest.mark.parametrize("stage", ["fit", "transform"])
def test_infinite_numeric_values_are_rejected(stage: str) -> None:
    transformer = WOETransformer(
        numeric_features=("value",), max_bins=2, min_bin_fraction=0.25
    )
    if stage == "fit":
        with pytest.raises(ValueError, match="infinite"):
            transformer.fit(pd.DataFrame({"value": [0.0, 1.0, np.inf, 2.0]}), [0, 1, 0, 1])
    else:
        transformer.fit(pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}), [0, 0, 1, 1])
        with pytest.raises(ValueError, match="infinite"):
            transformer.transform(pd.DataFrame({"value": [-np.inf]}))


@pytest.mark.parametrize("target", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_single_class_training_target_is_rejected(target: list[int]) -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        WOETransformer(
            numeric_features=("value",), max_bins=2, min_bin_fraction=0.25
        ).fit(pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}), target)


def test_transformer_is_cloneable_pipeline_compatible_and_has_stable_names() -> None:
    X = pd.DataFrame(
        {
            "category": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "value": np.arange(8, dtype=float),
        }
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    transformer = WOETransformer(
        categorical_features=("category",),
        numeric_features=("value",),
        max_bins=2,
        min_bin_fraction=0.125,
    )
    cloned = clone(transformer)
    pipeline = Pipeline(
        [("woe", cloned), ("classifier", LogisticRegression(random_state=42))]
    ).fit(X, y)

    assert pipeline.predict_proba(X).shape == (8, 2)
    assert pipeline.named_steps["woe"].get_feature_names_out().tolist() == [
        "category_woe",
        "value_woe",
    ]


def test_transform_needs_no_target_and_does_not_recompute_bins() -> None:
    X = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]})
    transformer = WOETransformer(
        numeric_features=("value",), max_bins=2, min_bin_fraction=0.25
    ).fit(X, [0, 0, 1, 1])
    original_table = transformer.binning_table_.copy(deep=True)

    first = transformer.transform(pd.DataFrame({"value": [0.5, 2.5]}))
    second = transformer.transform(pd.DataFrame({"value": [0.5, 2.5]}))

    assert list(inspect.signature(transformer.transform).parameters) == ["X"]
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(original_table, transformer.binning_table_)


def test_frozen_assignments_make_validation_and_test_labels_irrelevant(
    tmp_path: Path,
) -> None:
    rows = 40
    rng = np.random.default_rng(31)
    data: dict[str, np.ndarray] = {
        "customer_id": np.arange(1, rows + 1),
        "default_next_month": np.tile([0, 1], rows // 2),
    }
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.integers(0, 3, rows)
    for column in NUMERIC_COLUMNS:
        data[column] = rng.normal(size=rows)
    frame = pd.DataFrame(data)
    assignments = pd.DataFrame(
        {
            "customer_id": frame["customer_id"],
            "split": ["train"] * 20 + ["validation"] * 10 + ["test"] * 10,
        }
    )
    assignment_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignment_path, index=False, lineterminator="\n")
    digest = hashlib.sha256(assignment_path.read_bytes()).hexdigest()
    config = load_config("configs/experiment_v2.json")

    first = load_woe_development_data(
        frame, assignment_path, config=config, expected_assignment_sha256=digest
    )
    altered = frame.copy()
    altered.loc[20:, "default_next_month"] = 1 - altered.loc[20:, "default_next_month"]
    second = load_woe_development_data(
        altered, assignment_path, config=config, expected_assignment_sha256=digest
    )
    first_transformer = WOETransformer(
        categorical_features=config.categorical_features,
        numeric_features=config.numeric_features,
        max_bins=2,
        min_bin_fraction=0.1,
    ).fit(first.X_train, first.y_train)
    second_transformer = WOETransformer(
        categorical_features=config.categorical_features,
        numeric_features=config.numeric_features,
        max_bins=2,
        min_bin_fraction=0.1,
    ).fit(second.X_train, second.y_train)

    pd.testing.assert_frame_equal(
        first_transformer.binning_table_, second_transformer.binning_table_
    )
    assert first_transformer.iv_ == second_transformer.iv_


def test_report_is_complete_bilingual_and_uses_observed_values() -> None:
    X = pd.DataFrame({"value": [0.0, 0.0, 1.0, 1.0]})
    transformer = WOETransformer(
        numeric_features=("value",), max_bins=2, min_bin_fraction=0.25
    ).fit(X, [0, 0, 1, 1])

    report = format_woe_report(
        transformer,
        run_id="test_run",
        data_sha256="a" * 64,
        config_sha256="b" * 64,
        split_assignment_sha256="c" * 64,
        training_rows=4,
        validation_rows=2,
    )

    observed_iv = f"{transformer.iv_['value']:.6f}"
    assert report.startswith("# WOE 与 IV 分析报告（中文）")
    assert report.index("# WOE and IV Analysis Report (English)") > 0
    assert report.count(observed_iv) == 2
    assert report.count("test_run") == 2
    assert "不自动强制单调" in report
    assert "does not automatically force monotonicity" in report
