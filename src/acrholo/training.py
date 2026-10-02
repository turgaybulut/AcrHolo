import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from acrholo.data import Protein, split_training, validation_folds
from acrholo.model import FEATURE_DIMENSION
from acrholo.recipe import feature_recipe


def calibrate(scores: np.ndarray, labels: list[int]) -> tuple[float, float, float]:
    if (
        scores.ndim != 1
        or len(scores) != len(labels)
        or set(labels) != {0, 1}
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Calibration requires finite scores and both binary classes")
    pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        scores.reshape(-1, 1), labels
    )
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    slope = float(classifier.coef_[0, 0] / scaler.scale_[0])
    intercept = float(classifier.intercept_[0] - slope * scaler.mean_[0])
    positives = np.sort(expit(slope * scores[np.asarray(labels) == 1] + intercept))
    required = int(np.ceil(0.95 * len(positives)))
    threshold = float(min(positives[len(positives) - required], 0.95))
    if threshold < 0.05:
        raise ValueError("Reserved calibration cannot satisfy the fixed recall95 policy")
    return slope, intercept, threshold


def fit(values: np.ndarray, proteins: list[Protein], groups: list[str], seed: int = 42) -> dict:
    if (
        values.shape != (len(proteins), FEATURE_DIMENSION)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
    ):
        raise ValueError("Training requires matching finite FP64 feature rows")
    support, selection, reserve = split_training(proteins, groups, seed)
    development = sorted(support + selection)
    labels = [protein.label for protein in proteins]
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=2000, tol=1e-8)
    ).fit(values[development], [labels[index] for index in development])
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    weight = classifier.coef_[0] / scaler.scale_
    bias = classifier.intercept_[0] - weight @ scaler.mean_
    linear = {
        "weight": torch.tensor(weight, dtype=torch.float64),
        "bias": torch.tensor(bias, dtype=torch.float64),
    }
    scores = (torch.from_numpy(values[reserve]) @ linear["weight"] + linear["bias"]).numpy()
    return {
        "format": "acrholo-v1",
        "recipe": feature_recipe(),
        "linear": linear,
        "calibration": calibrate(scores, [labels[index] for index in reserve]),
        "training_n": len(development),
    }


def self_check() -> None:
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = [0, 0, 0, 1, 1, 1]
    slope, intercept, threshold = calibrate(scores, labels)
    probabilities = expit(slope * scores + intercept)
    assert 0.05 <= threshold <= 0.95 and np.mean(probabilities[3:] >= threshold) >= 0.95
    try:
        calibrate(np.zeros(2), [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("Calibration accepted a missing class")


def cross_validate(
    values: np.ndarray, proteins: list[Protein], groups: list[str], seed: int = 42
) -> list[dict]:
    if (
        values.shape != (len(proteins), FEATURE_DIMENSION)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
    ):
        raise ValueError("Cross-validation requires matching finite FP64 feature rows")
    support, selection, _ = split_training(proteins, groups, seed)
    development = sorted(support + selection)
    development_proteins = [proteins[index] for index in development]
    development_groups = [groups[index] for index in development]
    development_values = values[development]
    rows = []
    for fold, (training, held) in enumerate(
        validation_folds(development_proteins, development_groups, seed=seed)
    ):
        inner_fit, inner_calibration = validation_folds(
            [development_proteins[index] for index in training],
            [development_groups[index] for index in training],
            seed=seed,
        )[0]
        fit_indices = [training[index] for index in inner_fit]
        calibration_indices = [training[index] for index in inner_calibration]
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, class_weight="balanced", max_iter=2000, tol=1e-8),
        ).fit(
            development_values[fit_indices],
            [development_proteins[index].label for index in fit_indices],
        )
        combined_scores = classifier.decision_function(
            development_values[calibration_indices + held]
        )
        slope, intercept, threshold = calibrate(
            combined_scores[: len(calibration_indices)],
            [development_proteins[index].label for index in calibration_indices],
        )
        scores = combined_scores[len(calibration_indices) :]
        probabilities = expit(slope * scores + intercept)
        rows.extend(
            {
                "id": development_proteins[index].id,
                "fold": fold,
                "score": float(score),
                "probability": float(probability),
                "prediction": int(probability >= threshold),
            }
            for index, score, probability in zip(held, scores, probabilities, strict=True)
        )
    return rows
