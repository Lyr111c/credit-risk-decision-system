import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from credit_risk.modeling import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_model_pipelines,
    build_preprocessor,
    split_dataset,
)


def make_frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    target = np.tile([0, 0, 0, 1], rows // 4 + 1)[:rows]
    data: dict[str, np.ndarray] = {
        "customer_id": np.arange(1, rows + 1),
        "default_next_month": target,
    }
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.integers(0, 3, size=rows)
    for column in NUMERIC_COLUMNS:
        data[column] = rng.integers(1, 1_000, size=rows)
    return pd.DataFrame(data)


def test_feature_groups_are_complete_disjoint_and_exclude_non_features() -> None:
    assert len(FEATURE_COLUMNS) == 23
    assert set(CATEGORICAL_COLUMNS).isdisjoint(NUMERIC_COLUMNS)
    assert set(FEATURE_COLUMNS) == set(CATEGORICAL_COLUMNS) | set(NUMERIC_COLUMNS)
    assert "customer_id" not in FEATURE_COLUMNS
    assert "default_next_month" not in FEATURE_COLUMNS


def test_split_dataset_is_reproducible_complete_and_stratified() -> None:
    frame = make_frame()

    first = split_dataset(frame)
    second = split_dataset(frame)

    assert (len(first.X_train), len(first.X_validation), len(first.X_test)) == (60, 20, 20)
    assert first.X_train.index.equals(second.X_train.index)
    assert first.X_validation.index.equals(second.X_validation.index)
    assert first.X_test.index.equals(second.X_test.index)
    all_indices = set(first.X_train.index) | set(first.X_validation.index) | set(first.X_test.index)
    assert len(all_indices) == len(frame)
    assert set(first.X_train.index).isdisjoint(first.X_validation.index)
    assert set(first.X_train.index).isdisjoint(first.X_test.index)
    assert set(first.X_validation.index).isdisjoint(first.X_test.index)
    assert all(
        abs(part.mean() - frame["default_next_month"].mean()) < 0.01
        for part in (first.y_train, first.y_validation, first.y_test)
    )
    assert list(first.X_train.columns) == list(FEATURE_COLUMNS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0},
        {"validation_size": 0},
        {"test_size": 0.6, "validation_size": 0.4},
    ],
)
def test_split_dataset_rejects_invalid_proportions(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="split proportions"):
        split_dataset(make_frame(), **kwargs)


def test_split_dataset_reports_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns.*age"):
        split_dataset(make_frame().drop(columns="age"))


def test_preprocessor_handles_unknown_categories_without_changing_width() -> None:
    frame = make_frame(20)
    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(frame.loc[:, FEATURE_COLUMNS])
    unseen = frame.loc[[0], FEATURE_COLUMNS].copy()
    unseen.loc[:, "education"] = 99

    transformed_unseen = preprocessor.transform(unseen)

    assert isinstance(preprocessor, ColumnTransformer)
    assert transformed_unseen.shape == (1, transformed_train.shape[1])


def test_model_builders_return_three_pipelines_with_expected_weighting() -> None:
    models = build_model_pipelines()

    assert set(models) == {"dummy", "logistic", "logistic_balanced"}
    assert all(isinstance(model, Pipeline) for model in models.values())
    assert models["logistic"].named_steps["classifier"].class_weight is None
    assert models["logistic_balanced"].named_steps["classifier"].class_weight == "balanced"
    assert models["dummy"].named_steps["classifier"].strategy == "prior"
