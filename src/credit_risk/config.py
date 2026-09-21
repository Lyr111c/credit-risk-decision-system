"""Validated configuration for the version-2 credit-risk experiment protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from credit_risk.modeling import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the protocol schema."""


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float
    stratify: bool


@dataclass(frozen=True)
class CVConfig:
    folds: int
    shuffle: bool


@dataclass(frozen=True)
class WOEConfig:
    max_bins: int
    min_bin_fraction: float
    smoothing: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Small typed view over the fields used by the P0 implementation."""

    schema_version: int
    random_seed: int
    data_path: str
    data_sha256: str
    id_column: str
    target_column: str
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    split: SplitConfig
    cv: CVConfig
    woe: WOEConfig
    raw: dict[str, Any]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.categorical_features + self.numeric_features

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric")
    result = float(value)
    if not 0 < result < 1:
        raise ConfigError(f"{field} must be between 0 and 1")
    return result


def _positive_number(value: object, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric")
    result = float(value)
    if result < 0 or (result == 0 and not allow_zero):
        condition = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{field} must be {condition}")
    return result


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ConfigError(f"{field} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise ConfigError(f"{field} must not contain duplicates")
    return tuple(value)


def validate_config(raw: object) -> ExperimentConfig:
    """Validate and return the fields required to enforce the P0 protocol."""
    root = _mapping(raw, "config")
    data = _mapping(root.get("data"), "data")
    features = _mapping(root.get("features"), "features")
    split = _mapping(root.get("split"), "split")
    cv = _mapping(root.get("cross_validation"), "cross_validation")
    woe = _mapping(root.get("woe"), "woe")

    schema_version = _positive_int(root.get("schema_version"), "schema_version")
    if schema_version != 2:
        raise ConfigError("schema_version must be 2")
    random_seed = _positive_int(root.get("random_seed"), "random_seed", minimum=0)

    required_strings = {
        "data.path": data.get("path"),
        "data.sha256": data.get("sha256"),
        "data.id_column": data.get("id_column"),
        "data.target_column": data.get("target_column"),
    }
    for field, value in required_strings.items():
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{field} must be a non-empty string")
    data_sha256 = str(data["sha256"]).lower()
    if len(data_sha256) != 64 or any(c not in "0123456789abcdef" for c in data_sha256):
        raise ConfigError("data.sha256 must be a 64-character hexadecimal digest")
    if data["id_column"] != ID_COLUMN or data["target_column"] != TARGET_COLUMN:
        raise ConfigError("configured ID and target must match the modeling contract")

    categorical = _string_list(features.get("categorical"), "features.categorical")
    numeric = _string_list(features.get("numeric"), "features.numeric")
    if set(categorical) & set(numeric):
        raise ConfigError("categorical and numeric features must be disjoint")
    if categorical + numeric != FEATURE_COLUMNS:
        raise ConfigError("configured feature order must match the version-2 feature contract")
    if categorical != CATEGORICAL_COLUMNS or numeric != NUMERIC_COLUMNS:
        raise ConfigError("configured feature groups must match the version-2 feature contract")

    train = _probability(split.get("train"), "split.train")
    validation = _probability(split.get("validation"), "split.validation")
    test = _probability(split.get("test"), "split.test")
    if abs(train + validation + test - 1.0) > 1e-12:
        raise ConfigError("split proportions must sum to 1")
    if split.get("stratify") is not True:
        raise ConfigError("split.stratify must be true")

    folds = _positive_int(cv.get("folds"), "cross_validation.folds", minimum=2)
    if cv.get("shuffle") is not True:
        raise ConfigError("cross_validation.shuffle must be true")

    max_bins = _positive_int(woe.get("max_bins"), "woe.max_bins", minimum=1)
    min_bin_fraction = _positive_number(
        woe.get("min_bin_fraction"), "woe.min_bin_fraction"
    )
    if min_bin_fraction >= 1:
        raise ConfigError("woe.min_bin_fraction must be below 1")
    smoothing = _positive_number(woe.get("smoothing"), "woe.smoothing")

    search = _mapping(root.get("model_search"), "model_search")
    if search.get("primary_metric") != "roc_auc":
        raise ConfigError("model_search.primary_metric must be roc_auc")
    if search.get("secondary_metric") != "brier_score":
        raise ConfigError("model_search.secondary_metric must be brier_score")
    max_candidates = _positive_int(
        search.get("max_candidates"), "model_search.max_candidates"
    )
    if max_candidates > 12:
        raise ConfigError("model_search.max_candidates must not exceed 12")
    c_values = search.get("scorecard_c_values")
    if not isinstance(c_values, list) or not c_values:
        raise ConfigError("model_search.scorecard_c_values must be a non-empty list")
    for index, value in enumerate(c_values):
        _positive_number(value, f"model_search.scorecard_c_values[{index}]")

    score = _mapping(root.get("score_mapping"), "score_mapping")
    _positive_number(score.get("base_score"), "score_mapping.base_score")
    _positive_number(score.get("base_odds"), "score_mapping.base_odds")
    _positive_number(score.get("pdo"), "score_mapping.pdo")

    calibration = _mapping(root.get("calibration"), "calibration")
    if calibration.get("method") not in {"sigmoid", "none"}:
        raise ConfigError("calibration.method must be sigmoid or none")
    _positive_int(calibration.get("folds"), "calibration.folds", minimum=2)
    if not isinstance(calibration.get("isotonic_sensitivity"), bool):
        raise ConfigError("calibration.isotonic_sensitivity must be boolean")

    selection = _mapping(root.get("selection"), "selection")
    _positive_number(
        selection.get("roc_auc_tolerance"),
        "selection.roc_auc_tolerance",
        allow_zero=True,
    )
    _positive_number(
        selection.get("brier_tolerance"),
        "selection.brier_tolerance",
        allow_zero=True,
    )
    _string_list(selection.get("complexity_order"), "selection.complexity_order")

    strategy = _mapping(root.get("strategy"), "strategy")
    threshold_step = _positive_number(
        strategy.get("threshold_step"), "strategy.threshold_step"
    )
    if threshold_step > 1:
        raise ConfigError("strategy.threshold_step must be in (0, 1]")
    if strategy.get("approve_when") != "pd < threshold":
        raise ConfigError("strategy.approve_when must be 'pd < threshold'")
    _positive_number(strategy.get("primary_return"), "strategy.primary_return")
    _positive_number(strategy.get("primary_loss"), "strategy.primary_loss")

    duplicates = _mapping(root.get("duplicates"), "duplicates")
    if duplicates.get("main_protocol") != "retain":
        raise ConfigError("duplicates.main_protocol must be retain")
    if duplicates.get("grouped_split_sensitivity_only") is not True:
        raise ConfigError("duplicates.grouped_split_sensitivity_only must be true")

    metrics = _mapping(root.get("metrics"), "metrics")
    expected_metrics = {
        "average_precision_key": "average_precision",
        "legacy_pr_auc_alias": "pr_auc",
        "legacy_pr_auc_definition": "average_precision",
    }
    if any(metrics.get(key) != value for key, value in expected_metrics.items()):
        raise ConfigError("metrics must preserve the Average Precision naming contract")

    return ExperimentConfig(
        schema_version=schema_version,
        random_seed=random_seed,
        data_path=str(data["path"]),
        data_sha256=data_sha256,
        id_column=str(data["id_column"]),
        target_column=str(data["target_column"]),
        categorical_features=categorical,
        numeric_features=numeric,
        split=SplitConfig(train, validation, test, True),
        cv=CVConfig(folds, True),
        woe=WOEConfig(max_bins, min_bin_fraction, smoothing),
        raw=root,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load UTF-8 JSON and validate it against the version-2 contract."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Unable to load configuration {path}: {exc}") from exc
    return validate_config(raw)
