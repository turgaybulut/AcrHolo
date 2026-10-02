from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from acrholo.recipe import feature_recipe

FEATURE_DIMENSION = 12887


@torch.inference_mode()
def feature_matrix(
    embeddings: list[torch.Tensor],
    msa: np.ndarray,
    profiles: np.ndarray,
    available: np.ndarray,
    context: np.ndarray,
) -> np.ndarray:
    count = len(embeddings)
    if not count:
        raise ValueError("Provide a nonempty feature batch")
    for values, dimension in ((msa, 1536), (profiles, 1110), (context, 2560)):
        if (
            values.shape != (count, dimension)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Feature arrays must have matching rows, fixed dimensions and finite FP32 values"
            )
    if available.shape != (count,) or available.dtype != np.bool_:
        raise ValueError("Provide one boolean profile availability per query")
    blocks = (
        torch.from_numpy(np.column_stack((profiles, available)).astype(np.float32)),
        torch.from_numpy(msa),
        torch.from_numpy(context),
    )
    rows = []
    for index, embedding in enumerate(embeddings):
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.device.type != "cpu"
            or embedding.dtype != torch.float32
            or embedding.ndim != 2
            or embedding.shape[1] != 3840
            or not 1 <= embedding.shape[0] <= 4092
            or not torch.isfinite(embedding).all()
        ):
            raise ValueError("Embeddings must contain 1..4092 rows of 3840 finite CPU FP32 values")
        profiles_tensor, msa_tensor, context_tensor = blocks
        rows.append(
            torch.cat(
                (
                    embedding.mean(dim=0),
                    embedding.std(dim=0, correction=0),
                    msa_tensor[index],
                    profiles_tensor[index],
                    context_tensor[index],
                )
            )
        )
    return torch.stack(rows).numpy().astype(np.float64)


def validate_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("format") != "acrholo-v1" or checkpoint.get("recipe") != feature_recipe():
        raise ValueError("Checkpoint requires acrholo-v1 format and feature recipe")
    linear = checkpoint.get("linear")
    if not isinstance(linear, dict):
        raise ValueError("Checkpoint requires affine parameters")
    for name, shape in (("weight", (FEATURE_DIMENSION,)), ("bias", ())):
        tensor = linear.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float64
            or tensor.shape != shape
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Checkpoint requires finite CPU FP64 affine parameters")
    calibration = checkpoint.get("calibration")
    if (
        not isinstance(calibration, (tuple, list))
        or len(calibration) != 3
        or any(not isinstance(value, (int, float)) for value in calibration)
        or not np.isfinite(calibration).all()
        or not 0.05 <= calibration[2] <= 0.95
    ):
        raise ValueError("Checkpoint requires finite calibration and cutoff within [0.05, 0.95]")


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    validate_checkpoint(checkpoint)
    return checkpoint


@torch.inference_mode()
def predict(values: np.ndarray, checkpoint: dict) -> dict[str, np.ndarray]:
    validate_checkpoint(checkpoint)
    if (
        values.ndim != 2
        or values.shape[1] != FEATURE_DIMENSION
        or not len(values)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Prediction requires a nonempty finite FP64 feature matrix with 12887 columns"
        )
    linear = checkpoint["linear"]
    scores = (torch.from_numpy(values) @ linear["weight"] + linear["bias"]).numpy()
    slope, intercept, threshold = checkpoint["calibration"]
    probabilities = expit(slope * scores + intercept)
    return {
        "score": scores,
        "probability": probabilities,
        "prediction": (probabilities >= threshold).astype(np.int64),
    }


def self_check() -> None:
    embedding = torch.arange(7680, dtype=torch.float32).reshape(2, 3840)
    values = feature_matrix(
        [embedding],
        np.zeros((1, 1536), np.float32),
        np.zeros((1, 1110), np.float32),
        np.ones(1, np.bool_),
        np.zeros((1, 2560), np.float32),
    )
    assert values.shape == (1, FEATURE_DIMENSION) and values.dtype == np.float64
    assert values[0, 0] == 1920 and values[0, 3840] == 1920 and values[0, 10326] == 1
    checkpoint = {
        "format": "acrholo-v1",
        "recipe": feature_recipe(),
        "linear": {
            "weight": torch.zeros(FEATURE_DIMENSION, dtype=torch.float64),
            "bias": torch.tensor(0.0, dtype=torch.float64),
        },
        "calibration": (1.0, 0.0, 0.5),
    }
    result = predict(values, checkpoint)
    assert (
        result["score"][0] == 0 and result["probability"][0] == 0.5 and result["prediction"][0] == 1
    )
    for invalid in (values.astype(np.float32), np.full_like(values, np.nan), values[:, :-1]):
        try:
            predict(invalid, checkpoint)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid feature matrix accepted")
    try:
        validate_checkpoint({**checkpoint, "recipe": {}})
    except ValueError:
        pass
    else:
        raise AssertionError("Foreign feature recipe accepted")
