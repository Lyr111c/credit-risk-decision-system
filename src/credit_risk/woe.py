"""Auditable training-only binning, Weight of Evidence, and Information Value."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from credit_risk.config import ExperimentConfig, load_config
from credit_risk.data import load_dataset
from credit_risk.protocol import sha256_file


MISSING_LABEL = "MISSING"
OTHER_LABEL = "OTHER"
NEUTRAL_WOE = 0.0


@dataclass(frozen=True)
class WOEDevelopmentData:
    """Frozen training labels and label-free validation features for P1."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    split_assignment_sha256: str


def load_woe_development_data(
    frame: pd.DataFrame,
    assignments_path: str | Path,
    *,
    config: ExperimentConfig,
    expected_assignment_sha256: str,
) -> WOEDevelopmentData:
    """Join frozen IDs without selecting validation or test target values."""
    assignments_path = Path(assignments_path)
    actual_hash = sha256_file(assignments_path)
    if actual_hash != expected_assignment_sha256:
        raise ValueError(
            "Split-assignment SHA-256 mismatch: "
            f"expected {expected_assignment_sha256}, got {actual_hash}"
        )
    assignments = pd.read_csv(assignments_path)
    if list(assignments.columns) != [config.id_column, "split"]:
        raise ValueError("split assignments must contain customer_id and split columns")
    if assignments[config.id_column].duplicated().any():
        raise ValueError("split assignments contain duplicate customer IDs")
    if set(assignments["split"]) != {"train", "validation", "test"}:
        raise ValueError("split assignments must contain train, validation, and test")
    if set(assignments[config.id_column]) != set(frame[config.id_column]):
        raise ValueError("split assignments do not cover exactly the dataset customer IDs")

    split_by_id = assignments.set_index(config.id_column)["split"]
    row_splits = frame[config.id_column].map(split_by_id)
    train_mask = row_splits.eq("train")
    validation_mask = row_splits.eq("validation")
    return WOEDevelopmentData(
        X_train=frame.loc[train_mask, config.feature_columns],
        y_train=frame.loc[train_mask, config.target_column],
        X_validation=frame.loc[validation_mask, config.feature_columns],
        split_assignment_sha256=actual_hash,
    )


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _category_sort_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _format_bound(value: float) -> str:
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "+inf"
    return format(float(value), ".12g")


class WOETransformer(TransformerMixin, BaseEstimator):
    """Quantile/category binning followed by smoothed training-only WOE mapping.

    Numeric intervals are left-open and right-closed: ``(lower, upper]``.
    Missing values always use a separate learned bin when present during fit.
    Unknown categories use the learned OTHER bin, or neutral WOE zero when the
    training data did not create an OTHER bin.
    """

    def __init__(
        self,
        *,
        categorical_features: tuple[str, ...] = (),
        numeric_features: tuple[str, ...] = (),
        max_bins: int = 5,
        min_bin_fraction: float = 0.05,
        smoothing: float = 0.5,
    ) -> None:
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.max_bins = max_bins
        self.min_bin_fraction = min_bin_fraction
        self.smoothing = smoothing

    def _validate_parameters(self) -> None:
        if isinstance(self.max_bins, bool) or not isinstance(self.max_bins, int):
            raise ValueError("max_bins must be a positive integer")
        if self.max_bins < 1:
            raise ValueError("max_bins must be a positive integer")
        if not 0 < self.min_bin_fraction < 1:
            raise ValueError("min_bin_fraction must be between 0 and 1")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")
        categorical = tuple(self.categorical_features)
        numeric = tuple(self.numeric_features)
        if not categorical and not numeric:
            raise ValueError("at least one feature must be configured")
        if len(set(categorical + numeric)) != len(categorical) + len(numeric):
            raise ValueError("configured features must be unique and disjoint")

    def _validate_frame(self, X: object, *, fitting: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        configured = tuple(self.categorical_features) + tuple(self.numeric_features)
        missing = [column for column in configured if column not in X.columns]
        if missing:
            raise ValueError(f"Missing configured features: {missing}")
        if fitting:
            for column in self.numeric_features:
                if not is_numeric_dtype(X[column]):
                    raise ValueError(f"numeric feature {column} must have numeric dtype")
        for column in self.numeric_features:
            values = X[column].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"numeric feature {column} contains infinite values")
        return X.loc[:, configured]

    @staticmethod
    def _numeric_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.searchsorted(edges[1:-1], values, side="left")

    def _numeric_edges(self, series: pd.Series, min_count: int) -> np.ndarray:
        nonmissing = series.dropna().to_numpy(dtype=float)
        if len(nonmissing) == 0 or np.unique(nonmissing).size == 1:
            return np.array([-np.inf, np.inf], dtype=float)
        probabilities = np.linspace(0, 1, self.max_bins + 1)[1:-1]
        cuts = np.unique(np.quantile(nonmissing, probabilities))
        edges = np.concatenate(([-np.inf], cuts, [np.inf])).astype(float)
        while len(edges) > 2:
            indices = self._numeric_bin_indices(nonmissing, edges)
            counts = np.bincount(indices, minlength=len(edges) - 1)
            small = np.flatnonzero(counts < min_count)
            if len(small) == 0:
                break
            smallest_count = counts[small].min()
            index = int(small[counts[small] == smallest_count][0])
            if index == 0:
                boundary = 1
            elif index == len(counts) - 1:
                boundary = len(edges) - 2
            elif counts[index - 1] <= counts[index + 1]:
                boundary = index
            else:
                boundary = index + 1
            edges = np.delete(edges, boundary)
        return edges

    def _bin_statistics(
        self,
        feature: str,
        feature_type: str,
        bins: list[dict[str, Any]],
        y: np.ndarray,
    ) -> tuple[dict[int, float], list[dict[str, Any]], float]:
        good_total = int(np.sum(y == 0))
        bad_total = int(np.sum(y == 1))
        k = len(bins)
        woe_by_bin: dict[int, float] = {}
        records: list[dict[str, Any]] = []
        total_iv = 0.0
        for order, item in enumerate(bins):
            mask = item["mask"]
            count = int(mask.sum())
            good = int(np.sum(y[mask] == 0))
            bad = int(np.sum(y[mask] == 1))
            p_good = (good + self.smoothing) / (good_total + self.smoothing * k)
            p_bad = (bad + self.smoothing) / (bad_total + self.smoothing * k)
            woe = float(np.log(p_good / p_bad))
            iv_component = float((p_good - p_bad) * woe)
            total_iv += iv_component
            bin_id = int(item["bin_id"])
            woe_by_bin[bin_id] = woe
            records.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    "bin_id": item["display_id"],
                    "bin_order": order,
                    "bin_label": item["label"],
                    "lower_bound": item.get("lower_bound"),
                    "upper_bound": item.get("upper_bound"),
                    "right_closed": item.get("right_closed"),
                    "categories": item.get("categories"),
                    "sample_count": count,
                    "good_count": good,
                    "bad_count": bad,
                    "default_rate": bad / count,
                    "p_good": p_good,
                    "p_bad": p_bad,
                    "woe": woe,
                    "iv_component": iv_component,
                }
            )
        for record in records:
            record["total_iv"] = total_iv
        return woe_by_bin, records, total_iv

    def _fit_numeric(
        self, feature: str, series: pd.Series, y: np.ndarray, min_count: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
        edges = self._numeric_edges(series, min_count)
        nonmissing = ~series.isna().to_numpy()
        values = series.to_numpy(dtype=float)
        indices = np.full(len(series), -1, dtype=int)
        indices[nonmissing] = self._numeric_bin_indices(values[nonmissing], edges)
        bins: list[dict[str, Any]] = []
        if nonmissing.any():
            for index, (lower, upper) in enumerate(
                zip(edges[:-1], edges[1:], strict=True)
            ):
                bins.append(
                    {
                        "bin_id": index,
                        "display_id": f"bin_{index}",
                        "label": f"({_format_bound(lower)}, {_format_bound(upper)}]",
                        "lower_bound": float(lower),
                        "upper_bound": float(upper),
                        "right_closed": True,
                        "categories": None,
                        "mask": indices == index,
                    }
                )
        missing_bin: int | None = None
        if (~nonmissing).any():
            missing_bin = len(bins)
            bins.append(
                {
                    "bin_id": missing_bin,
                    "display_id": "missing",
                    "label": MISSING_LABEL,
                    "lower_bound": None,
                    "upper_bound": None,
                    "right_closed": None,
                    "categories": None,
                    "mask": ~nonmissing,
                }
            )
        woe_by_bin, records, total_iv = self._bin_statistics(
            feature, "numeric", bins, y
        )
        return (
            {
                "type": "numeric",
                "edges": edges,
                "woe_by_bin": woe_by_bin,
                "missing_bin": missing_bin,
            },
            records,
            total_iv,
        )

    def _fit_categorical(
        self, feature: str, series: pd.Series, y: np.ndarray, min_count: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
        missing_mask = series.isna().to_numpy()
        counts = series.loc[~series.isna()].value_counts(dropna=False)
        frequent = sorted(
            [value for value, count in counts.items() if count >= min_count],
            key=_category_sort_key,
        )
        infrequent = sorted(
            [value for value, count in counts.items() if count < min_count],
            key=_category_sort_key,
        )
        category_to_bin: dict[Any, int] = {}
        bins: list[dict[str, Any]] = []
        for value in frequent:
            bin_id = len(bins)
            category_to_bin[value] = bin_id
            python_value = _python_scalar(value)
            bins.append(
                {
                    "bin_id": bin_id,
                    "display_id": f"category_{bin_id}",
                    "label": repr(python_value),
                    "categories": json.dumps([python_value], ensure_ascii=False),
                    "mask": (series == value).fillna(False).to_numpy(),
                }
            )
        other_bin: int | None = None
        if infrequent:
            other_bin = len(bins)
            for value in infrequent:
                category_to_bin[value] = other_bin
            python_values = [_python_scalar(value) for value in infrequent]
            bins.append(
                {
                    "bin_id": other_bin,
                    "display_id": "other",
                    "label": OTHER_LABEL,
                    "categories": json.dumps(python_values, ensure_ascii=False),
                    "mask": series.isin(infrequent).to_numpy(),
                }
            )
        missing_bin: int | None = None
        if missing_mask.any():
            missing_bin = len(bins)
            bins.append(
                {
                    "bin_id": missing_bin,
                    "display_id": "missing",
                    "label": MISSING_LABEL,
                    "categories": None,
                    "mask": missing_mask,
                }
            )
        woe_by_bin, records, total_iv = self._bin_statistics(
            feature, "categorical", bins, y
        )
        return (
            {
                "type": "categorical",
                "category_to_bin": category_to_bin,
                "woe_by_bin": woe_by_bin,
                "other_bin": other_bin,
                "missing_bin": missing_bin,
            },
            records,
            total_iv,
        )

    def fit(self, X: object, y: object) -> "WOETransformer":
        """Learn bins and WOE values from one training fold."""
        self._validate_parameters()
        frame = self._validate_frame(X, fitting=True)
        if len(frame) == 0:
            raise ValueError("X and y must not be empty")
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(frame):
            raise ValueError("y must be one-dimensional and match X length")
        if pd.isna(target).any() or set(np.unique(target)) != {0, 1}:
            raise ValueError("y must contain both binary classes 0 and 1")

        min_count = max(1, math.ceil(self.min_bin_fraction * len(frame)))
        specs: dict[str, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []
        iv: dict[str, float] = {}
        for feature in self.categorical_features:
            spec, feature_records, total_iv = self._fit_categorical(
                feature, frame[feature], target, min_count
            )
            specs[feature] = spec
            records.extend(feature_records)
            iv[feature] = total_iv
        for feature in self.numeric_features:
            spec, feature_records, total_iv = self._fit_numeric(
                feature, frame[feature], target, min_count
            )
            specs[feature] = spec
            records.extend(feature_records)
            iv[feature] = total_iv

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        self.feature_names_out_ = np.asarray(
            [f"{column}_woe" for column in frame.columns], dtype=object
        )
        self.bin_specs_ = specs
        self.iv_ = iv
        self.binning_table_ = pd.DataFrame.from_records(records)
        return self

    def transform(self, X: object) -> pd.DataFrame:
        """Apply frozen bins without labels or distribution re-estimation."""
        check_is_fitted(self, ("bin_specs_", "binning_table_", "iv_"))
        frame = self._validate_frame(X, fitting=False)
        transformed: dict[str, np.ndarray] = {}
        for feature in frame.columns:
            spec = self.bin_specs_[feature]
            series = frame[feature]
            values = np.full(len(frame), NEUTRAL_WOE, dtype=float)
            missing = series.isna().to_numpy()
            if spec["type"] == "numeric":
                numeric = series.to_numpy(dtype=float)
                indices = self._numeric_bin_indices(
                    numeric[~missing], spec["edges"]
                )
                values[~missing] = [
                    spec["woe_by_bin"].get(int(index), NEUTRAL_WOE)
                    for index in indices
                ]
            else:
                for row, value in enumerate(series.to_numpy(dtype=object)):
                    if missing[row]:
                        continue
                    bin_id = spec["category_to_bin"].get(value, spec["other_bin"])
                    if bin_id is not None:
                        values[row] = spec["woe_by_bin"][bin_id]
            if missing.any() and spec["missing_bin"] is not None:
                values[missing] = spec["woe_by_bin"][spec["missing_bin"]]
            transformed[f"{feature}_woe"] = values
        return pd.DataFrame(transformed, index=frame.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and tuple(input_features) != tuple(
            self.feature_names_in_
        ):
            raise ValueError("input_features must match the fitted feature order")
        return self.feature_names_out_.copy()


def _numeric_trend(group: pd.DataFrame) -> str:
    ordered = group.loc[group["bin_id"] != "missing"].sort_values("bin_order")
    values = ordered["woe"].to_numpy(dtype=float)
    if len(values) <= 1:
        return "single_bin"
    differences = np.diff(values)
    tolerance = 1e-12
    if np.all(np.abs(differences) <= tolerance):
        return "constant"
    if np.all(differences >= -tolerance):
        return "increasing"
    if np.all(differences <= tolerance):
        return "decreasing"
    return "non_monotonic"


def woe_summary(transformer: WOETransformer) -> pd.DataFrame:
    """Return one auditable IV and numeric-trend row per configured feature."""
    check_is_fitted(transformer, "binning_table_")
    rows = []
    for feature in transformer.feature_names_in_:
        group = transformer.binning_table_.loc[
            transformer.binning_table_["feature"] == feature
        ]
        feature_type = str(group["feature_type"].iloc[0])
        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "bin_count": len(group),
                "total_iv": float(group["total_iv"].iloc[0]),
                "woe_trend": (
                    _numeric_trend(group) if feature_type == "numeric" else "not_applicable"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("total_iv", ascending=False, kind="stable")


def _markdown_summary(summary: pd.DataFrame, *, english: bool) -> str:
    trend_zh = {
        "single_bin": "单箱",
        "constant": "常量",
        "increasing": "递增",
        "decreasing": "递减",
        "non_monotonic": "非单调",
        "not_applicable": "不适用",
    }
    trend_en = {
        "single_bin": "single bin",
        "constant": "constant",
        "increasing": "increasing",
        "decreasing": "decreasing",
        "non_monotonic": "non-monotonic",
        "not_applicable": "not applicable",
    }
    headers = (
        ["Feature", "Type", "Bins", "Total IV", "Numeric WOE trend"]
        if english
        else ["变量", "类型", "箱数", "总 IV", "数值变量 WOE 趋势"]
    )
    rows = []
    for row in summary.itertuples(index=False):
        trend = trend_en[row.woe_trend] if english else trend_zh[row.woe_trend]
        feature_type = (
            row.feature_type
            if english
            else {"numeric": "数值", "categorical": "类别"}[row.feature_type]
        )
        rows.append(
            [row.feature, feature_type, str(row.bin_count), f"{row.total_iv:.6f}", trend]
        )
    table = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(items) + " |" for items in table)


def format_woe_report(
    transformer: WOETransformer,
    *,
    run_id: str,
    data_sha256: str,
    config_sha256: str,
    split_assignment_sha256: str,
    training_rows: int,
    validation_rows: int,
) -> str:
    """Render the actual P1 fit as a Chinese-first bilingual report."""
    summary = woe_summary(transformer)
    non_monotonic = int((summary["woe_trend"] == "non_monotonic").sum())
    table_zh = _markdown_summary(summary, english=False)
    table_en = _markdown_summary(summary, english=True)
    return f"""# WOE 与 IV 分析报告（中文）

运行 ID：`{run_id}`。本报告由 `credit_risk.woe` 根据固定训练集实际拟合生成，不使用验证集或测试集标签。

## 数据与方法

- 原始数据 SHA-256：`{data_sha256}`；配置 SHA-256：`{config_sha256}`；客户切分清单 SHA-256：`{split_assignment_sha256}`。
- 拟合样本为固定训练集 {training_rows:,} 条。验证集 {validation_rows:,} 条只执行无标签转换和有限值检查；验证/测试标签均未提供给转换器或用于分箱、WOE、IV 计算。
- 数值变量首先按训练集分位数产生最多 `{transformer.max_bins}` 个箱，重复切点合并；相邻小箱按训练期频数合并，直到每个非缺失箱至少占训练样本 `{transformer.min_bin_fraction:.0%}`，或只剩一个箱。区间采用左开右闭 `(lower, upper]`，并覆盖负无穷至正无穷。
- 类别变量中低于 `{transformer.min_bin_fraction:.0%}` 的训练期水平合并为 `OTHER`。缺失值保持独立箱。未见类别优先映射到训练期 `OTHER`；若没有 `OTHER`，使用中性 `WOE=0` 并保留该回退规则。
- 定义违约为 bad=1、未违约为 good=0，平滑参数 `α={transformer.smoothing}`。`WOE=ln(p_good/p_bad)`，IV 为各箱 `(p_good-p_bad)×WOE` 之和。全 good 或全 bad 的箱通过平滑得到有限值。

## 变量结果

{table_zh}

机器可读分箱表记录每箱边界或训练期类别、样本数、good/bad 数、违约率、WOE、IV 分量与变量总 IV。本阶段不依据固定 IV 阈值删除变量；如未来做 IV 筛选，必须放在交叉验证训练折内。

14 个数值变量中有 {non_monotonic} 个呈非单调 WOE 序列。该检查只描述训练期分箱，不自动强制单调，也不把相关性解释为因果关系。来源未定义的还款状态编码仍作为独立类别或按纯频数进入 `OTHER`，没有赋予业务含义。

## 限制

IV 来自同一固定训练样本，可能高估样本外稳定性；随机切分不是 OOT。WOE/IV 仅用于解释和后续评分卡候选，不代表合规拒绝理由或行业统一筛选标准。P2 必须在交叉验证折内重新拟合本转换器，不能复用全训练集拟合结果进行折内评分。

---

# WOE and IV Analysis Report (English)

Run ID: `{run_id}`. This report was generated by fitting `credit_risk.woe` on the frozen training set. It uses neither validation nor test labels.

## Data and method

- Raw-data SHA-256: `{data_sha256}`; configuration SHA-256: `{config_sha256}`; customer split-manifest SHA-256: `{split_assignment_sha256}`.
- The fit uses {training_rows:,} frozen training observations. The {validation_rows:,} validation observations undergo only label-free transformation and a finite-value check; validation/test labels are neither supplied to the transformer nor used for binning, WOE, or IV.
- Numeric features begin with at most `{transformer.max_bins}` training-quantile bins, merging repeated cut points. Adjacent sparse bins are merged by training frequency until every non-missing bin contains at least `{transformer.min_bin_fraction:.0%}` of training observations or only one bin remains. Intervals are left-open and right-closed `(lower, upper]` and cover negative to positive infinity.
- Training categories below `{transformer.min_bin_fraction:.0%}` are combined as `OTHER`. Missing values retain a separate bin. An unseen category uses the learned `OTHER` bin when present; otherwise it receives neutral `WOE=0`, with this fallback preserved.
- Default is bad=1 and non-default is good=0, with smoothing `α={transformer.smoothing}`. `WOE=ln(p_good/p_bad)`, and IV is the sum of `(p_good-p_bad)×WOE` across bins. Smoothing keeps all-good and all-bad bins finite.

## Feature results

{table_en}

The machine-readable bin table records every boundary or training category, sample count, good/bad count, default rate, WOE, IV component, and feature-total IV. This stage does not remove features using fixed IV cutoffs. Any future IV selection must occur inside each cross-validation training fold.

Among 14 numeric features, {non_monotonic} have non-monotonic WOE sequences. This is a training-bin description only: the implementation does not automatically force monotonicity or interpret association as causation. Undocumented repayment-status codes remain separate categories or enter `OTHER` by frequency alone; no business meaning is invented.

## Limitations

IV is estimated from the same frozen training sample and may overstate out-of-sample stability; the random split is not OOT. WOE/IV supports interpretation and the later scorecard candidate, but it is neither a compliant adverse-action reason nor a universal industry screening rule. P2 must refit this transformer inside each cross-validation fold rather than reuse the full-training fit for fold-level scoring.
"""


def materialize_woe_analysis(
    config_path: str | Path,
    *,
    run_id: str,
    output_root: str | Path = "outputs",
    assignments_path: str | Path = "outputs/artifacts/p0_protocol_20260921/split_assignments.csv",
    protocol_manifest_path: str | Path = "outputs/artifacts/p0_protocol_20260921/split_manifest.json",
) -> dict[str, Path]:
    """Fit P1 on training labels and write aggregate artifacts and report."""
    config: ExperimentConfig = load_config(config_path)
    data_path = Path(config.data_path)
    data_sha256 = sha256_file(data_path)
    if data_sha256 != config.data_sha256:
        raise ValueError(
            f"Data SHA-256 mismatch: expected {config.data_sha256}, got {data_sha256}"
        )
    frame = load_dataset(data_path)
    protocol_manifest = json.loads(
        Path(protocol_manifest_path).read_text(encoding="utf-8")
    )
    if protocol_manifest.get("config_sha256") != config.sha256:
        raise ValueError("P0 protocol manifest configuration hash is stale")
    development = load_woe_development_data(
        frame,
        assignments_path,
        config=config,
        expected_assignment_sha256=protocol_manifest["split_assignment_sha256"],
    )
    transformer = WOETransformer(
        categorical_features=config.categorical_features,
        numeric_features=config.numeric_features,
        max_bins=config.woe.max_bins,
        min_bin_fraction=config.woe.min_bin_fraction,
        smoothing=config.woe.smoothing,
    ).fit(development.X_train, development.y_train)
    validation_transformed = transformer.transform(development.X_validation)
    if not np.isfinite(validation_transformed.to_numpy()).all():
        raise RuntimeError("validation transformation produced non-finite WOE values")

    output_root = Path(output_root)
    artifact_directory = output_root / "artifacts" / run_id
    report_directory = output_root / "reports"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "binning_table": artifact_directory / "woe_binning_table.csv",
        "metadata": artifact_directory / "woe_metadata.json",
        "report": report_directory / "woe_iv_report.md",
    }
    transformer.binning_table_.to_csv(paths["binning_table"], index=False)
    table_hash = hashlib.sha256(paths["binning_table"].read_bytes()).hexdigest()
    metadata = {
        "run_id": run_id,
        "data_sha256": data_sha256,
        "config_sha256": config.sha256,
        "split_assignment_sha256": development.split_assignment_sha256,
        "training_rows": len(development.X_train),
        "validation_transform_rows": len(validation_transformed),
        "labels_used_for_fit": ["training"],
        "validation_labels_used": False,
        "test_labels_used": False,
        "parameters": {
            "max_bins": transformer.max_bins,
            "min_bin_fraction": transformer.min_bin_fraction,
            "smoothing": transformer.smoothing,
            "numeric_interval": "(lower, upper]",
            "unseen_category_fallback": "OTHER if learned, else WOE=0",
        },
        "feature_iv": transformer.iv_,
        "binning_table_sha256": table_hash,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(
        format_woe_report(
            transformer,
            run_id=run_id,
            data_sha256=data_sha256,
            config_sha256=config.sha256,
            split_assignment_sha256=development.split_assignment_sha256,
            training_rows=len(development.X_train),
            validation_rows=len(validation_transformed),
        ),
        encoding="utf-8",
    )
    return paths


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_v2.json"))
    parser.add_argument("--run-id", default="p1_woe_20260921")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=Path("outputs/artifacts/p0_protocol_20260921/split_assignments.csv"),
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("outputs/artifacts/p0_protocol_20260921/split_manifest.json"),
    )
    args = parser.parse_args()
    for name, path in materialize_woe_analysis(
        args.config,
        run_id=args.run_id,
        output_root=args.output_root,
        assignments_path=args.split_assignments,
        protocol_manifest_path=args.protocol_manifest,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    _main()
