"""Reproducible splitting, preprocessing, and baseline model construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
LOGISTIC_MAX_ITER = 2_000
TARGET_COLUMN = "default_next_month"
ID_COLUMN = "customer_id"

CATEGORICAL_COLUMNS = (
    "sex",
    "education",
    "marital_status",
    "repayment_status_sep",
    "repayment_status_aug",
    "repayment_status_jul",
    "repayment_status_jun",
    "repayment_status_may",
    "repayment_status_apr",
)
NUMERIC_COLUMNS = (
    "credit_limit",
    "age",
    "bill_amount_sep",
    "bill_amount_aug",
    "bill_amount_jul",
    "bill_amount_jun",
    "bill_amount_may",
    "bill_amount_apr",
    "payment_amount_sep",
    "payment_amount_aug",
    "payment_amount_jul",
    "payment_amount_jun",
    "payment_amount_may",
    "payment_amount_apr",
)
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


@dataclass(frozen=True)
class DataSplits:
    """Feature and target partitions created from one stratified random split."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> DataSplits:
    """Create reproducible target-stratified train, validation, and test sets."""
    if (
        test_size <= 0
        or validation_size <= 0
        or test_size + validation_size >= 1
    ):
        raise ValueError("split proportions must be positive and sum to less than 1")

    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    target_values = set(frame[TARGET_COLUMN].dropna().unique())
    if target_values != {0, 1}:
        raise ValueError("target must contain both binary values 0 and 1")

    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_development,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build training-fitted transformations for the fixed baseline features."""
    return ColumnTransformer(
        transformers=(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        )
    )


def build_model_pipelines() -> dict[str, Pipeline]:
    """Build the dummy and logistic-regression candidates with isolated preprocessors."""
    classifiers = {
        "dummy": DummyClassifier(strategy="prior", random_state=RANDOM_SEED),
        "logistic": LogisticRegression(
            random_state=RANDOM_SEED, max_iter=LOGISTIC_MAX_ITER
        ),
        "logistic_balanced": LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=LOGISTIC_MAX_ITER,
            class_weight="balanced",
        ),
    }
    return {
        name: Pipeline(
            steps=(
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            )
        )
        for name, classifier in classifiers.items()
    }
