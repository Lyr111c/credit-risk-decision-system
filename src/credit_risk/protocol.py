"""Freeze, persist, and report the version-2 experiment protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from credit_risk.config import ExperimentConfig, load_config
from credit_risk.data import load_dataset
from credit_risk.modeling import FEATURE_COLUMNS, split_dataset


@dataclass(frozen=True)
class DevelopmentData:
    """Only the data that development and model-selection code may consume."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series


@dataclass(frozen=True)
class FinalEvaluationData:
    """Test data exposed only to an explicit final-evaluation call."""

    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass(frozen=True)
class ProtocolSplit:
    development: DevelopmentData
    final_evaluation: FinalEvaluationData
    assignments: pd.DataFrame
    assignment_sha256: str


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assignment_bytes(assignments: pd.DataFrame) -> bytes:
    ordered = assignments.sort_values("customer_id", kind="stable")
    return ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")


def create_protocol_split(
    frame: pd.DataFrame, config: ExperimentConfig
) -> ProtocolSplit:
    """Create the fixed split and isolate test labels from development data."""
    if config.id_column not in frame or not frame[config.id_column].is_unique:
        raise ValueError(f"{config.id_column} must exist and be unique")
    if tuple(config.feature_columns) != FEATURE_COLUMNS:
        raise ValueError("configuration feature contract differs from modeling contract")

    split = split_dataset(
        frame,
        test_size=config.split.test,
        validation_size=config.split.validation,
        random_state=config.random_seed,
    )
    split_by_index: dict[Any, str] = {}
    for name, indices in (
        ("train", split.X_train.index),
        ("validation", split.X_validation.index),
        ("test", split.X_test.index),
    ):
        for index in indices:
            if index in split_by_index:
                raise ValueError("source frame index must be unique")
            split_by_index[index] = name
    if len(split_by_index) != len(frame):
        raise ValueError("split assignments are not complete")

    assignments = pd.DataFrame(
        {
            "customer_id": frame[config.id_column],
            "split": frame.index.map(split_by_index),
        }
    )
    assignment_sha256 = hashlib.sha256(_assignment_bytes(assignments)).hexdigest()
    return ProtocolSplit(
        development=DevelopmentData(
            split.X_train,
            split.y_train,
            split.X_validation,
            split.y_validation,
        ),
        final_evaluation=FinalEvaluationData(split.X_test, split.y_test),
        assignments=assignments,
        assignment_sha256=assignment_sha256,
    )


def build_stratified_cv(config: ExperimentConfig) -> StratifiedKFold:
    """Build the shared shuffled CV object used with full sklearn pipelines."""
    return StratifiedKFold(
        n_splits=config.cv.folds,
        shuffle=config.cv.shuffle,
        random_state=config.random_seed,
    )


def duplicate_split_summary(
    frame: pd.DataFrame, assignments: pd.DataFrame, *, id_column: str
) -> dict[str, int]:
    """Summarize retained non-ID duplicates and whether groups cross splits."""
    labeled = frame.drop(columns=id_column).copy()
    labeled["__split"] = assignments.set_index("customer_id").loc[
        frame[id_column], "split"
    ].to_numpy()
    content_columns = [column for column in labeled if column != "__split"]
    duplicate_mask = labeled.duplicated(content_columns, keep=False)
    duplicate_rows = int(labeled.duplicated(content_columns).sum())
    groups = labeled.loc[duplicate_mask].groupby(content_columns, dropna=False, sort=False)
    duplicate_groups = 0
    cross_split_groups = 0
    cross_split_rows = 0
    for _, group in groups:
        duplicate_groups += 1
        if group["__split"].nunique() > 1:
            cross_split_groups += 1
            cross_split_rows += len(group)
    return {
        "duplicate_rows_after_first": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "cross_split_groups": cross_split_groups,
        "rows_in_cross_split_groups": cross_split_rows,
    }


def _git_value(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip() or "clean"
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({type(exc).__name__})"


def format_environment_snapshot(config: ExperimentConfig, data_sha256: str) -> str:
    """Return a compact exact environment snapshot for the protocol run."""
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pytest", "xlrd")
    lines = [
        f"python={platform.python_version()}",
        f"python_executable={sys.executable}",
        f"platform={platform.platform()}",
        *(f"{name}={importlib.metadata.version(name)}" for name in packages),
        f"data_sha256={data_sha256}",
        f"config_sha256={config.sha256}",
        f"git_commit={_git_value(['rev-parse', 'HEAD'])}",
        "git_status_begin",
        _git_value(["status", "--short"]),
        "git_status_end",
    ]
    return "\n".join(lines) + "\n"


def protocol_manifest(
    config: ExperimentConfig,
    split: ProtocolSplit,
    duplicate_summary: dict[str, int],
    *,
    data_sha256: str,
    run_id: str,
) -> dict[str, object]:
    assignments = split.assignments
    counts = assignments["split"].value_counts()
    target_rates = {
        "train": float(split.development.y_train.mean()),
        "validation": float(split.development.y_validation.mean()),
        "test": float(split.final_evaluation.y_test.mean()),
    }
    return {
        "run_id": run_id,
        "schema_version": config.schema_version,
        "random_seed": config.random_seed,
        "data_sha256": data_sha256,
        "config_sha256": config.sha256,
        "split_assignment_sha256": split.assignment_sha256,
        "split_counts": {name: int(counts[name]) for name in ("train", "validation", "test")},
        "target_rates": target_rates,
        "cv": {"folds": config.cv.folds, "shuffle": config.cv.shuffle},
        "features": list(config.feature_columns),
        "id_is_feature": config.id_column in config.feature_columns,
        "test_label_policy": "explicit_final_evaluation_only",
        "average_precision": {
            "canonical_key": "average_precision",
            "legacy_alias": "pr_auc",
            "legacy_alias_definition": "average_precision_score",
        },
        "duplicates": duplicate_summary,
    }


def format_protocol_report(manifest: dict[str, object]) -> str:
    """Render a complete Chinese-first bilingual P0 protocol report."""
    counts = manifest["split_counts"]
    rates = manifest["target_rates"]
    duplicates = manifest["duplicates"]
    return f"""# 实验协议报告（中文）

运行 ID：`{manifest['run_id']}`。本报告由 `credit_risk.protocol` 根据实际数据与配置生成。

## 固定数据与切分

- 原始数据 SHA-256：`{manifest['data_sha256']}`。
- 配置 SHA-256：`{manifest['config_sha256']}`。
- 本地客户切分清单 SHA-256：`{manifest['split_assignment_sha256']}`。清单含客户 ID，已被 Git 精确忽略；仓库仅保留本汇总与哈希。
- 随机种子为 `{manifest['random_seed']}`，采用目标分层的 60%/20%/20% 随机切分：训练集 {counts['train']:,} 条（违约率 {rates['train']:.4%}）、验证集 {counts['validation']:,} 条（违约率 {rates['validation']:.4%}）、测试集 {counts['test']:,} 条（违约率 {rates['test']:.4%}）。三组客户 ID 互斥且覆盖全部记录。
- `customer_id` 只用于关联、完整性检查和切分清单，不属于 {len(manifest['features'])} 个模型特征。

## 评估边界

新的开发入口只接收训练集和验证集；其结果对象不包含测试特征、测试标签或测试预测。测试标签仅能通过显式最终评估入口使用。现有 `run_baseline_experiment` 作为历史报告复现接口保留，不能用于后续搜索或模型选择。

训练阶段默认使用 `{manifest['cv']['folds']}` 折、打乱且固定种子的分层交叉验证。预处理器与估计器必须作为同一个 scikit-learn `Pipeline` 进入每一折，避免在全训练集上预先拟合预处理。

验证集用于开发期候选比较；测试集留到 P6，在模型、校准方式和策略均冻结后显式评估。历史报告已经展示过同一测试集，因此它不是“从未查看”的测试集，也不是 OOT；本协议不能恢复独立的时间外证据。

## 指标定义

规范名称 `average_precision` 使用 `average_precision_score`。旧字段 `pr_auc` 为兼容历史报告而保留，但它只是 Average Precision（AP）的别名，不是梯形积分 PR-AUC。未来若增加梯形积分指标，必须使用不同字段名。

## 重复记录规则

去掉 ID 后共有 {duplicates['duplicate_rows_after_first']} 条重复记录（按每组第一条之后计数），涉及 {duplicates['duplicate_groups']} 个重复内容组。其中 {duplicates['cross_split_groups']} 个组跨越切分，共包含 {duplicates['rows_in_cross_split_groups']} 条记录。主协议保留全部记录以维持历史基准；不删除、不分组合并。分组切分只能作为单独标记的敏感性实验，不能替换主比较。

## 限制与后续依赖

这是既有客户行为风险数据，没有可靠时间轴，随机切分不能证明跨时期稳定。P1 及后续阶段必须读取同一配置与切分哈希，并继续遵守开发/最终评估边界。本阶段没有重新选择模型、校准概率、制定业务策略或运行最终测试评估。

---

# Experiment Protocol Report (English)

Run ID: `{manifest['run_id']}`. This report was generated by `credit_risk.protocol` from the actual data and configuration.

## Frozen data and split

- Raw-data SHA-256: `{manifest['data_sha256']}`.
- Configuration SHA-256: `{manifest['config_sha256']}`.
- Local customer split-manifest SHA-256: `{manifest['split_assignment_sha256']}`. The manifest contains customer IDs and is precisely ignored by Git; only this aggregate summary and hash are retained in the repository.
- Seed `{manifest['random_seed']}` creates a target-stratified 60%/20%/20% random split: {counts['train']:,} training rows (default rate {rates['train']:.4%}), {counts['validation']:,} validation rows (default rate {rates['validation']:.4%}), and {counts['test']:,} test rows (default rate {rates['test']:.4%}). Customer IDs are mutually exclusive across the three complete partitions.
- `customer_id` is used only for linkage, integrity checks, and the split manifest; it is not one of the {len(manifest['features'])} model features.

## Evaluation boundary

The new development entry point receives only training and validation data. Its result contains no test features, labels, or predictions. Test labels are available only through an explicit final-evaluation entry point. The existing `run_baseline_experiment` is retained solely to reproduce the historical report and must not be called by later search or model-selection workflows.

Training defaults to `{manifest['cv']['folds']}` shuffled stratified folds with a fixed seed. Each preprocessor and estimator must enter every fold together as one scikit-learn `Pipeline`, preventing preprocessing from being pre-fitted on the full training set.

Validation supports development-time candidate comparison. Test data are reserved for explicit P6 evaluation after the model, calibration, and policy are frozen. The historical report has already exposed this same test set, so it is neither previously unseen nor OOT; this protocol cannot recreate independent temporal evidence.

## Metric definition

The canonical `average_precision` metric uses `average_precision_score`. The old `pr_auc` key remains as a compatibility alias for historical reports, but it means Average Precision (AP), not trapezoidal PR-AUC. Any future trapezoidal metric must use a distinct field name.

## Duplicate-record rule

After removing ID, there are {duplicates['duplicate_rows_after_first']} duplicate rows counted after each group's first occurrence, across {duplicates['duplicate_groups']} duplicate-content groups. Of these, {duplicates['cross_split_groups']} groups cross split boundaries and contain {duplicates['rows_in_cross_split_groups']} rows. The main protocol retains every record to preserve the historical baseline; it neither drops nor group-merges them. A grouped-split analysis may be run only as a separately labeled sensitivity study and cannot replace the primary comparison.

## Limitations and downstream dependency

These are existing-customer behavioral-risk data without a reliable timeline, so a random split cannot establish temporal stability. P1 and later stages must use the same configuration and split hash and preserve the development/final-evaluation boundary. This stage did not reselect models, calibrate probabilities, define a business policy, or run final test evaluation.
"""


def materialize_protocol(
    config_path: str | Path,
    *,
    run_id: str,
    output_root: str | Path = "outputs",
) -> dict[str, Path]:
    """Validate inputs and write the P0 local artifacts and bilingual report."""
    config = load_config(config_path)
    data_path = Path(config.data_path)
    actual_hash = sha256_file(data_path)
    if actual_hash != config.data_sha256:
        raise ValueError(
            f"Data SHA-256 mismatch: expected {config.data_sha256}, got {actual_hash}"
        )
    frame = load_dataset(data_path)
    split = create_protocol_split(frame, config)
    duplicates = duplicate_split_summary(frame, split.assignments, id_column=config.id_column)
    manifest = protocol_manifest(
        config, split, duplicates, data_sha256=actual_hash, run_id=run_id
    )

    output_root = Path(output_root)
    artifact_directory = output_root / "artifacts" / run_id
    report_directory = output_root / "reports"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "assignments": artifact_directory / "split_assignments.csv",
        "manifest": artifact_directory / "split_manifest.json",
        "environment": artifact_directory / "environment.txt",
        "report": report_directory / "experiment_protocol.md",
    }
    paths["assignments"].write_bytes(_assignment_bytes(split.assignments))
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["environment"].write_text(
        format_environment_snapshot(config, actual_hash), encoding="utf-8"
    )
    paths["report"].write_text(format_protocol_report(manifest), encoding="utf-8")
    return paths


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_v2.json"))
    parser.add_argument("--run-id", default="p0_protocol_20260921")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    for name, path in materialize_protocol(
        args.config, run_id=args.run_id, output_root=args.output_root
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    _main()
