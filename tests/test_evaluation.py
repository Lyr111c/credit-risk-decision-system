import pytest

from credit_risk.evaluation import evaluate_binary_classifier


def test_evaluate_binary_classifier_returns_hand_checkable_metrics() -> None:
    result = evaluate_binary_classifier(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.35, 0.8],
        threshold=0.5,
        calibration_bins=2,
    )

    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["pr_auc"] == pytest.approx(5 / 6)
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["pr_auc"] == result["average_precision"]
    assert result["ks"] == pytest.approx(0.5)
    assert result["brier_score"] == pytest.approx(0.158125)
    assert result["threshold"] == 0.5
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert result["calibration"] == [
        {
            "mean_predicted_probability": pytest.approx(0.2833333333),
            "observed_default_rate": pytest.approx(1 / 3),
            "count": 3,
        },
        {
            "mean_predicted_probability": pytest.approx(0.8),
            "observed_default_rate": pytest.approx(1.0),
            "count": 1,
        },
    ]


@pytest.mark.parametrize(
    ("target", "probability", "message"),
    [
        ([0, 1], [0.2], "same length"),
        ([0, 2], [0.2, 0.8], "binary"),
        ([0, 1], [-0.1, 0.8], "between 0 and 1"),
        ([0, 1], [0.2, 1.1], "between 0 and 1"),
    ],
)
def test_evaluate_binary_classifier_rejects_invalid_inputs(
    target: list[int], probability: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_binary_classifier(target, probability)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_evaluate_binary_classifier_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate_binary_classifier([0, 1], [0.2, 0.8], threshold=threshold)


def test_evaluate_binary_classifier_requires_both_target_classes() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        evaluate_binary_classifier([0, 0], [0.2, 0.3])


def test_calibration_counts_follow_sklearn_boundary_assignment() -> None:
    result = evaluate_binary_classifier(
        [0, 0, 1, 1],
        [0.1, 0.5, 0.6, 0.9],
        calibration_bins=2,
    )

    assert [point["count"] for point in result["calibration"]] == [2, 2]
