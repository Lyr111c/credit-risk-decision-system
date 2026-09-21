"""Training-CV WOE logistic scorecard with an auditable points mapping."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from credit_risk.config import ExperimentConfig, load_config
from credit_risk.data import load_dataset
from credit_risk.evaluation import evaluate_binary_classifier
from credit_risk.modeling import LOGISTIC_MAX_ITER, build_model_pipelines
from credit_risk.protocol import build_stratified_cv, sha256_file
from credit_risk.woe import WOETransformer, load_woe_development_data


@dataclass(frozen=True)
class ScoreMapping:
    """Traditional score scaling for good:bad odds."""

    base_score: float = 600.0
    base_odds: float = 20.0
    pdo: float = 50.0

    def __post_init__(self) -> None:
        if self.base_odds <= 0 or self.pdo <= 0:
            raise ValueError("base_odds and pdo must be positive")

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def probability_to_score(self, probability: Any) -> float | np.ndarray:
        values = np.asarray(probability, dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("probability must contain finite values in [0, 1]")
        epsilon = np.finfo(float).eps
        clipped = np.clip(values, epsilon, 1 - epsilon)
        logit = np.log(clipped / (1 - clipped))
        scores = self.offset - self.factor * logit
        return float(scores) if scores.ndim == 0 else scores

    def score_to_probability(self, score: Any) -> float | np.ndarray:
        values = np.asarray(score, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("score must contain finite values")
        probabilities = expit((self.offset - values) / self.factor)
        return float(probabilities) if probabilities.ndim == 0 else probabilities


@dataclass(frozen=True)
class ScorecardDevelopmentData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    split_assignment_sha256: str


@dataclass(frozen=True)
class ScorecardSelection:
    selected_c: float
    fold_results: pd.DataFrame
    summary: pd.DataFrame


class FittedScorecard:
    """Fitted WOE-logistic pipeline plus exact additive score decomposition."""

    def __init__(self, pipeline: Pipeline, mapping: ScoreMapping, selected_c: float):
        self.pipeline = pipeline
        self.mapping = mapping
        self.selected_c = selected_c
        self._validate_pipeline()

    def _validate_pipeline(self) -> None:
        if tuple(self.pipeline.named_steps) != ("woe", "classifier"):
            raise ValueError("pipeline must contain woe then classifier")
        transformer = self.pipeline.named_steps["woe"]
        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(transformer, "binning_table_") or not hasattr(classifier, "coef_"):
            raise ValueError("scorecard pipeline must be fitted")
        if classifier.coef_.shape != (1, len(transformer.feature_names_in_)):
            raise ValueError("scorecard requires one binary-logistic coefficient per feature")

    @property
    def transformer(self) -> WOETransformer:
        return self.pipeline.named_steps["woe"]

    @property
    def classifier(self) -> LogisticRegression:
        return self.pipeline.named_steps["classifier"]

    @property
    def base_points(self) -> float:
        return float(
            self.mapping.offset - self.mapping.factor * self.classifier.intercept_[0]
        )

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.mapping.probability_to_score(self.predict_pd(X)))

    def score_components(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transformer.transform(X)
        coefficients = self.classifier.coef_[0]
        result = pd.DataFrame(index=X.index)
        result["base_points"] = self.base_points
        contribution_columns = []
        for index, feature in enumerate(self.transformer.feature_names_in_):
            column = f"{feature}_points"
            contribution_columns.append(column)
            result[column] = (
                -self.mapping.factor
                * coefficients[index]
                * transformed[f"{feature}_woe"].to_numpy()
            )
        result["total_score"] = result["base_points"] + result[
            contribution_columns
        ].sum(axis=1)
        result["raw_pd"] = self.predict_pd(X)
        return result

    def points_table(self) -> pd.DataFrame:
        table = self.transformer.binning_table_.copy()
        coefficient_by_feature = dict(
            zip(self.transformer.feature_names_in_, self.classifier.coef_[0], strict=True)
        )
        table["coefficient"] = table["feature"].map(coefficient_by_feature)
        table["points"] = (
            -self.mapping.factor * table["coefficient"] * table["woe"]
        )
        return table

    def reason_codes(self, X: pd.DataFrame, *, top_n: int = 3) -> list[list[dict[str, Any]]]:
        """Return largest point deductions from each feature's best learned bin."""
        if top_n < 1:
            raise ValueError("top_n must be positive")
        components = self.score_components(X)
        points = self.points_table()
        reference = points.groupby("feature")["points"].max().to_dict()
        output: list[list[dict[str, Any]]] = []
        for _, row in components.iterrows():
            reasons = []
            for feature in self.transformer.feature_names_in_:
                actual = float(row[f"{feature}_points"])
                deduction = actual - max(float(reference[feature]), 0.0)
                if deduction < -1e-12:
                    reasons.append(
                        {
                            "feature": str(feature),
                            "deduction_points": deduction,
                            "actual_points": actual,
                        }
                    )
            reasons.sort(key=lambda item: item["deduction_points"])
            output.append(reasons[:top_n])
        return output


@dataclass(frozen=True)
class ScorecardDevelopmentResult:
    scorecard: FittedScorecard
    selection: ScorecardSelection
    baseline_model: Pipeline
    validation_probabilities: dict[str, np.ndarray]
    validation_metrics: dict[str, dict[str, object]]
    validation_scores: np.ndarray


def build_scorecard_pipeline(config: ExperimentConfig, *, c_value: float) -> Pipeline:
    """Build an unfitted fold-safe WOE and unweighted logistic pipeline."""
    return Pipeline(
        [
            (
                "woe",
                WOETransformer(
                    categorical_features=config.categorical_features,
                    numeric_features=config.numeric_features,
                    max_bins=config.woe.max_bins,
                    min_bin_fraction=config.woe.min_bin_fraction,
                    smoothing=config.woe.smoothing,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=None,
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def select_scorecard_c(
    X_train: pd.DataFrame, y_train: pd.Series, config: ExperimentConfig
) -> ScorecardSelection:
    """Select C using training folds only; exact ties prefer smaller C."""
    records: list[dict[str, float | int]] = []
    cv = build_stratified_cv(config)
    for c_value in config.model_search.scorecard_c_values:
        template = build_scorecard_pipeline(config, c_value=c_value)
        for fold, (fit_indices, score_indices) in enumerate(
            cv.split(X_train, y_train), start=1
        ):
            model = clone(template)
            start = time.perf_counter()
            model.fit(X_train.iloc[fit_indices], y_train.iloc[fit_indices])
            probability = model.predict_proba(X_train.iloc[score_indices])[:, 1]
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "c_value": float(c_value),
                    "fold": fold,
                    "roc_auc": float(roc_auc_score(y_train.iloc[score_indices], probability)),
                    "brier_score": float(
                        brier_score_loss(y_train.iloc[score_indices], probability)
                    ),
                    "fit_score_seconds": elapsed,
                    "fit_rows": len(fit_indices),
                    "score_rows": len(score_indices),
                }
            )
    fold_results = pd.DataFrame(records)
    summary = (
        fold_results.groupby("c_value", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_brier_score=("brier_score", "mean"),
            std_brier_score=("brier_score", "std"),
            total_seconds=("fit_score_seconds", "sum"),
        )
        .sort_values("c_value")
        .reset_index(drop=True)
    )
    selected = summary.sort_values(
        ["mean_roc_auc", "mean_brier_score", "c_value"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    return ScorecardSelection(float(selected["c_value"]), fold_results, summary)


def load_scorecard_development_data(
    frame: pd.DataFrame,
    assignments_path: str | Path,
    *,
    config: ExperimentConfig,
    expected_assignment_sha256: str,
) -> ScorecardDevelopmentData:
    """Load train/validation data from fixed IDs without selecting test labels."""
    woe_data = load_woe_development_data(
        frame,
        assignments_path,
        config=config,
        expected_assignment_sha256=expected_assignment_sha256,
    )
    return ScorecardDevelopmentData(
        X_train=woe_data.X_train,
        y_train=woe_data.y_train,
        X_validation=woe_data.X_validation,
        y_validation=frame.loc[woe_data.X_validation.index, config.target_column],
        split_assignment_sha256=woe_data.split_assignment_sha256,
    )


def fit_scorecard_development(
    data: ScorecardDevelopmentData, config: ExperimentConfig
) -> ScorecardDevelopmentResult:
    """Select and fit the scorecard, then compare on validation only."""
    selection = select_scorecard_c(data.X_train, data.y_train, config)
    pipeline = build_scorecard_pipeline(config, c_value=selection.selected_c)
    pipeline.fit(data.X_train, data.y_train)
    mapping = ScoreMapping(
        config.score_mapping.base_score,
        config.score_mapping.base_odds,
        config.score_mapping.pdo,
    )
    scorecard = FittedScorecard(pipeline, mapping, selection.selected_c)
    scorecard_probability = scorecard.predict_pd(data.X_validation)

    baseline = build_model_pipelines()["logistic"]
    baseline.fit(data.X_train, data.y_train)
    baseline_probability = baseline.predict_proba(data.X_validation)[:, 1]
    probabilities = {
        "ordinary_logistic": baseline_probability,
        "woe_scorecard": scorecard_probability,
    }
    metrics = {
        name: evaluate_binary_classifier(data.y_validation, probability)
        for name, probability in probabilities.items()
    }
    return ScorecardDevelopmentResult(
        scorecard=scorecard,
        selection=selection,
        baseline_model=baseline,
        validation_probabilities=probabilities,
        validation_metrics=metrics,
        validation_scores=scorecard.score(data.X_validation),
    )


def score_decile_table(
    scores: np.ndarray, probabilities: np.ndarray, target: pd.Series
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"score": scores, "raw_pd": probabilities, "target": target.to_numpy()}
    )
    frame["score_decile"] = pd.qcut(frame["score"], q=10, duplicates="drop")
    table = (
        frame.groupby("score_decile", observed=True)
        .agg(
            sample_count=("target", "size"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            mean_raw_pd=("raw_pd", "mean"),
            observed_default_rate=("target", "mean"),
        )
        .reset_index(drop=True)
    )
    table.insert(0, "score_decile", np.arange(1, len(table) + 1))
    return table


def generate_scorecard_figure(
    result: ScorecardDevelopmentResult,
    target: pd.Series,
    output_path: str | Path,
) -> Path:
    """Generate one bilingual validation score-distribution diagnostic."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deciles = score_decile_table(
        result.validation_scores,
        result.validation_probabilities["woe_scorecard"],
        target,
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(result.validation_scores, bins=30, color="#4C78A8", edgecolor="white")
    axes[0].set_title("Validation score distribution / 验证集分数分布")
    axes[0].set_xlabel("Raw score / 原始评分卡分数")
    axes[0].set_ylabel("Count / 样本数")
    axes[1].plot(
        deciles["score_decile"],
        deciles["observed_default_rate"],
        marker="o",
        label="Observed / 实际",
    )
    axes[1].plot(
        deciles["score_decile"],
        deciles["mean_raw_pd"],
        marker="s",
        label="Mean raw PD / 平均原始 PD",
    )
    axes[1].set_title("Default rate by score decile / 分数十分位违约率")
    axes[1].set_xlabel("Score decile: low to high / 分数十分位：低到高")
    axes[1].set_ylabel("Rate / 比例")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _markdown_cv(summary: pd.DataFrame) -> str:
    rows = [
        [
            f"{row.c_value:g}",
            f"{row.mean_roc_auc:.6f}",
            f"{row.std_roc_auc:.6f}",
            f"{row.mean_brier_score:.6f}",
            f"{row.total_seconds:.2f}",
        ]
        for row in summary.itertuples(index=False)
    ]
    table = [
        ["C", "Mean ROC-AUC", "ROC-AUC SD", "Mean Brier", "Total seconds"],
        ["---"] * 5,
        *rows,
    ]
    return "\n".join("| " + " | ".join(row) + " |" for row in table)


def _markdown_metrics(metrics: dict[str, dict[str, object]], *, english: bool) -> str:
    names = (
        {
            "ordinary_logistic": "Ordinary logistic regression",
            "woe_scorecard": "WOE scorecard",
        }
        if english
        else {"ordinary_logistic": "普通逻辑回归", "woe_scorecard": "WOE 评分卡"}
    )
    rows = []
    for name in ("ordinary_logistic", "woe_scorecard"):
        item = metrics[name]
        rows.append(
            [
                names[name],
                f"{item['roc_auc']:.6f}",
                f"{item['average_precision']:.6f}",
                f"{item['ks']:.6f}",
                f"{item['brier_score']:.6f}",
            ]
        )
    headers = (
        ["Model", "ROC-AUC", "Average Precision", "KS", "Brier Score"]
        if english
        else ["模型", "ROC-AUC", "Average Precision", "KS", "Brier Score"]
    )
    table = [headers, ["---"] * 5, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table)


def _markdown_deciles(table: pd.DataFrame, *, english: bool) -> str:
    headers = (
        ["Score decile", "N", "Min score", "Max score", "Mean raw PD", "Default rate"]
        if english
        else ["分数十分位", "样本数", "最低分", "最高分", "平均原始 PD", "违约率"]
    )
    rows = [
        [
            str(int(row.score_decile)),
            str(int(row.sample_count)),
            f"{row.min_score:.2f}",
            f"{row.max_score:.2f}",
            f"{row.mean_raw_pd:.4%}",
            f"{row.observed_default_rate:.4%}",
        ]
        for row in table.itertuples(index=False)
    ]
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def format_scorecard_report(
    result: ScorecardDevelopmentResult,
    data: ScorecardDevelopmentData,
    *,
    run_id: str,
    data_sha256: str,
    config_sha256: str,
) -> str:
    scorecard = result.scorecard
    deciles = score_decile_table(
        result.validation_scores,
        result.validation_probabilities["woe_scorecard"],
        data.y_validation,
    )
    cv_table = _markdown_cv(result.selection.summary)
    metrics_zh = _markdown_metrics(result.validation_metrics, english=False)
    metrics_en = _markdown_metrics(result.validation_metrics, english=True)
    deciles_zh = _markdown_deciles(deciles, english=False)
    deciles_en = _markdown_deciles(deciles, english=True)
    scores = result.validation_scores
    mapping = scorecard.mapping
    return f"""# 评分卡开发报告（中文）

运行 ID：`{run_id}`。本报告由 `credit_risk.scorecard` 实际执行生成。模型开发只使用固定训练集和验证集；测试标签未加载到评分卡开发数据对象中。

## 训练与参数选择

评分卡采用 `WOETransformer → LogisticRegression`，不使用类别权重。`C ∈ {list(result.selection.summary['c_value'])}` 仅在 {len(data.X_train):,} 条训练数据内部进行 5 折分层交叉验证；每折重新拟合分箱、WOE 和逻辑回归。主指标是平均 ROC-AUC，其次是平均 Brier Score，完全并列时选择更小的 `C`。实际选中 `C={result.selection.selected_c:g}`。

{cv_table}

数据 SHA-256：`{data_sha256}`；配置 SHA-256：`{config_sha256}`；切分清单 SHA-256：`{data.split_assignment_sha256}`。

## 分数映射

本项目采用教学映射：`base_score={mapping.base_score:g}`、`base_odds={mapping.base_odds:g}`（good:bad）、`PDO={mapping.pdo:g}`。它们不是行业统一标准。`B=PDO/ln(2)={mapping.factor:.6f}`，`A=base_score-B×ln(base_odds)={mapping.offset:.6f}`。

对原始评分卡 PD，`score=A-B×logit(PD)`；反解为 `PD=1/(1+exp((score-A)/B))`。20:1 的 good:bad odds 对应 600 分，good:bad odds 每翻倍一次增加 50 分。模型基础分为 `A-B×β₀={scorecard.base_points:.6f}`，每个变量的箱分为 `-B×βⱼ×WOEⱼ`。内部计算保留浮点精度，报告展示才取整或格式化。

## 验证集比较

{metrics_zh}

该比较使用 {len(data.X_validation):,} 条验证记录，不读取测试结果，也不据此重新分箱。评分卡与普通逻辑回归的差异是开发期描述，不代表统计显著性或最终模型选择。

![评分卡验证诊断](../figures/scorecard_validation.png)

## 分数分段

{deciles_zh}

验证集原始分数范围为 {scores.min():.2f} 至 {scores.max():.2f}，中位数为 {np.median(scores):.2f}。分数越高表示原始模型 PD 越低。逐客户解释以各变量训练期最高箱分为参考，报告主要减分项；这些是模型内解释，不是因果结论或合规拒绝理由。

## 限制与后续使用

当前分数只与未校准的评分卡原始 PD 精确对应。P4 若对 PD 做校准，必须分别保存 `raw_score`、原始 PD 和校准后决策 PD，不能声称该加性分数仍精确对应校准后 PD。数据是既有客户行为数据，随机验证集不是 OOT；本阶段不冻结最终模型或审批策略。

---

# Scorecard Development Report (English)

Run ID: `{run_id}`. This report was generated by executing `credit_risk.scorecard`. Model development uses only the frozen training and validation sets; test labels are not loaded into the scorecard-development data object.

## Training and parameter selection

The scorecard uses `WOETransformer → LogisticRegression` without class weights. `C ∈ {list(result.selection.summary['c_value'])}` is evaluated only through five stratified folds inside the {len(data.X_train):,} training observations; binning, WOE, and logistic regression are refitted in every fold. Mean ROC-AUC is primary, mean Brier Score is secondary, and an exact tie prefers smaller `C`. The selected value is `C={result.selection.selected_c:g}`.

{cv_table}

Data SHA-256: `{data_sha256}`; configuration SHA-256: `{config_sha256}`; split-manifest SHA-256: `{data.split_assignment_sha256}`.

## Score mapping

The educational mapping is `base_score={mapping.base_score:g}`, `base_odds={mapping.base_odds:g}` good:bad, and `PDO={mapping.pdo:g}`. These are not universal industry standards. `B=PDO/ln(2)={mapping.factor:.6f}` and `A=base_score-B×ln(base_odds)={mapping.offset:.6f}`.

For the raw scorecard PD, `score=A-B×logit(PD)` and `PD=1/(1+exp((score-A)/B))`. Good:bad odds of 20:1 map to 600, and every doubling of good:bad odds adds 50 points. Base points are `A-B×β₀={scorecard.base_points:.6f}`; each feature-bin contribution is `-B×βⱼ×WOEⱼ`. Internal calculations retain floating-point precision and rounding occurs only for display.

## Validation comparison

{metrics_en}

This comparison uses {len(data.X_validation):,} validation observations. It reads no test results and does not re-bin after validation. Differences between the scorecard and ordinary logistic regression are development descriptions, not significance claims or final model selection.

![Scorecard validation diagnostics](../figures/scorecard_validation.png)

## Score bands

{deciles_en}

Validation raw scores range from {scores.min():.2f} to {scores.max():.2f}, with median {np.median(scores):.2f}. Higher scores mean lower raw-model PD. Customer-level explanations compare each contribution with the feature's highest training-bin points and report the largest deductions; these are model explanations, not causal findings or compliant adverse-action reasons.

## Limitations and downstream use

The score currently maps exactly only to the uncalibrated raw scorecard PD. If P4 calibrates PD, `raw_score`, raw PD, and calibrated decision PD must be recorded separately; the additive score must not be claimed to map exactly to calibrated PD. These are existing-customer behavioral data and the random validation set is not OOT. This stage does not freeze a final model or approval policy.
"""


def materialize_scorecard(
    config_path: str | Path,
    *,
    run_id: str,
    output_root: str | Path = "outputs",
    assignments_path: str | Path = "outputs/artifacts/p0_protocol_20260921/split_assignments.csv",
    protocol_manifest_path: str | Path = "outputs/artifacts/p0_protocol_20260921/split_manifest.json",
) -> dict[str, Path]:
    config = load_config(config_path)
    data_path = Path(config.data_path)
    data_sha256 = sha256_file(data_path)
    if data_sha256 != config.data_sha256:
        raise ValueError(
            f"Data SHA-256 mismatch: expected {config.data_sha256}, got {data_sha256}"
        )
    protocol_manifest = json.loads(
        Path(protocol_manifest_path).read_text(encoding="utf-8")
    )
    if protocol_manifest.get("config_sha256") != config.sha256:
        raise ValueError("P0 protocol manifest configuration hash is stale")
    frame = load_dataset(data_path)
    data = load_scorecard_development_data(
        frame,
        assignments_path,
        config=config,
        expected_assignment_sha256=protocol_manifest["split_assignment_sha256"],
    )
    result = fit_scorecard_development(data, config)

    output_root = Path(output_root)
    artifact_directory = output_root / "artifacts" / run_id
    report_directory = output_root / "reports"
    figure_directory = output_root / "figures"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "cv_results": artifact_directory / "scorecard_cv_results.csv",
        "points_table": artifact_directory / "scorecard_points.csv",
        "metadata": artifact_directory / "scorecard_metadata.json",
        "figure": figure_directory / "scorecard_validation.png",
        "report": report_directory / "scorecard_report.md",
    }
    result.selection.fold_results.to_csv(paths["cv_results"], index=False)
    result.scorecard.points_table().to_csv(paths["points_table"], index=False)
    generate_scorecard_figure(result, data.y_validation, paths["figure"])
    metadata = {
        "run_id": run_id,
        "data_sha256": data_sha256,
        "config_sha256": config.sha256,
        "split_assignment_sha256": data.split_assignment_sha256,
        "training_rows": len(data.X_train),
        "validation_rows": len(data.X_validation),
        "test_labels_used": False,
        "selected_c": result.selection.selected_c,
        "mapping": {
            "base_score": result.scorecard.mapping.base_score,
            "base_odds": result.scorecard.mapping.base_odds,
            "pdo": result.scorecard.mapping.pdo,
            "factor_b": result.scorecard.mapping.factor,
            "offset_a": result.scorecard.mapping.offset,
            "base_points": result.scorecard.base_points,
        },
        "intercept": float(result.scorecard.classifier.intercept_[0]),
        "coefficients": dict(
            zip(
                result.scorecard.transformer.feature_names_in_,
                map(float, result.scorecard.classifier.coef_[0]),
                strict=True,
            )
        ),
        "validation_metrics": result.validation_metrics,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(
        format_scorecard_report(
            result,
            data,
            run_id=run_id,
            data_sha256=data_sha256,
            config_sha256=config.sha256,
        ),
        encoding="utf-8",
    )
    return paths


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_v2.json"))
    parser.add_argument("--run-id", default="p2_scorecard_20260921")
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
    for name, path in materialize_scorecard(
        args.config,
        run_id=args.run_id,
        output_root=args.output_root,
        assignments_path=args.split_assignments,
        protocol_manifest_path=args.protocol_manifest,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    _main()
