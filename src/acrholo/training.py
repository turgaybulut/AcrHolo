from collections.abc import Iterator

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from acrholo.data import Protein, split_training, validation_folds
from acrholo.model import FEATURE_DIMENSION
from acrholo.recipe import feature_recipe


def calibrate(
    scores: np.ndarray, labels: list[int], *, allow_invalid: bool = False
) -> tuple[float, float, float]:
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
        if not allow_invalid:
            raise ValueError("Reserved calibration cannot satisfy the fixed recall95 policy")
        threshold = float("nan")
    return slope, intercept, threshold


def fit_classifier(values: np.ndarray, labels: list[int]) -> Pipeline:
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=2000, tol=1e-8)
    ).fit(values, labels)
    classifier = pipeline.steps[-1][1]
    if int(classifier.n_iter_.max()) >= classifier.max_iter:
        raise ValueError(f"Logistic solver reached {classifier.max_iter} iterations")
    return pipeline


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
    pipeline = fit_classifier(values[development], [labels[index] for index in development])
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
    invalid_scores = np.r_[np.full(100, -3.0), np.full(2, -3.0), np.full(18, 3.0)]
    invalid_labels = [0] * 100 + [1] * 20
    invalid_slope, invalid_intercept, invalid_threshold = calibrate(
        invalid_scores, invalid_labels, allow_invalid=True
    )
    assert np.isfinite([invalid_slope, invalid_intercept]).all() and np.isnan(invalid_threshold)
    try:
        calibrate(invalid_scores, invalid_labels)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid recall95 policy accepted in strict calibration")
    proteins = [Protein(f"train_{index:04d}", "AX", index % 2) for index in range(100)]
    groups = [protein.id for protein in proteins]
    sliced = np.asarray([[protein.label] for protein in proteins], dtype=np.float64)
    folds = list(cross_validation_folds(sliced, proteins, groups))
    assert len(folds) == 5 and sum(len(held) for _, held, _ in folds) == 80
    support, selection, reserve = split_training(proteins, groups)
    expected = {proteins[index].id for index in support + selection}
    assert {row["id"] for _, held, _ in folds for row in held} == expected
    reserved = {proteins[index].id for index in reserve}
    for metadata, held, calibration in folds:
        assert metadata["dimension"] == 1 and metadata["policy_valid"]
        assert metadata["solver_converged"] and metadata["fit_n"] == metadata["train_n"]
        assert not {row["id"] for row in held} & {row["id"] for row in calibration}
        assert not reserved & {row["id"] for row in held + calibration}
        assert all(row["decision"] == row["label"] for row in held)
    try:
        calibrate(np.zeros(2), [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("Calibration accepted a missing class")


def cross_validation_folds(
    values: np.ndarray,
    proteins: list[Protein],
    groups: list[str],
    seed: int = 42,
    *,
    allow_invalid: bool = False,
) -> Iterator[tuple[dict, list[dict], list[dict]]]:
    if (
        values.ndim != 2
        or values.shape[0] != len(proteins)
        or not len(proteins)
        or not 1 <= values.shape[1] <= FEATURE_DIMENSION
        or values.dtype != np.float64
        or not np.isfinite(values).all()
    ):
        raise ValueError("Cross-validation requires matching nonempty finite FP64 feature rows")
    support, selection, _ = split_training(proteins, groups, seed)
    development = sorted(support + selection)
    development_proteins = [proteins[index] for index in development]
    development_groups = [groups[index] for index in development]
    development_values = values[development]
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
        classifier = fit_classifier(
            development_values[fit_indices],
            [development_proteins[index].label for index in fit_indices],
        )
        combined_scores = classifier.decision_function(
            development_values[calibration_indices + held]
        )
        calibration_labels = [development_proteins[index].label for index in calibration_indices]
        slope, intercept, threshold = calibrate(
            combined_scores[: len(calibration_indices)],
            calibration_labels,
            allow_invalid=allow_invalid,
        )
        probabilities = expit(slope * combined_scores + intercept)
        valid = bool(np.isfinite(threshold))
        combined_rows = [
            {
                "fold": fold,
                "id": development_proteins[index].id,
                "family": development_groups[index],
                "label": development_proteins[index].label,
                "score": float(score),
                "probability": float(probability),
                "threshold": threshold,
                "policy_valid": valid,
                "decision": int(probability >= threshold) if valid else float("nan"),
            }
            for index, score, probability in zip(
                calibration_indices + held, combined_scores, probabilities, strict=True
            )
        ]
        head = classifier.steps[-1][1]
        metadata = {
            "fold": fold,
            "dimension": values.shape[1],
            "train_n": len(fit_indices),
            "fit_n": len(fit_indices),
            "solver_iterations": int(head.n_iter_.max()),
            "solver_converged": True,
            "calibration_n": len(calibration_indices),
            "held_n": len(held),
            "held_positives": sum(development_proteins[index].label for index in held),
            "calibration_positives": sum(calibration_labels),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "threshold": threshold,
            "policy_valid": valid,
            "policy": "recall95-continuous",
        }
        yield (
            metadata,
            combined_rows[len(calibration_indices) :],
            combined_rows[: len(calibration_indices)],
        )


def cross_validate(
    values: np.ndarray, proteins: list[Protein], groups: list[str], seed: int = 42
) -> list[dict]:
    if (
        values.shape != (len(proteins), FEATURE_DIMENSION)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
    ):
        raise ValueError("Cross-validation requires matching finite FP64 feature rows")
    return [
        {
            "id": row["id"],
            "fold": row["fold"],
            "score": row["score"],
            "probability": row["probability"],
            "prediction": row["decision"],
        }
        for _, held_rows, _ in cross_validation_folds(values, proteins, groups, seed)
        for row in held_rows
    ]
