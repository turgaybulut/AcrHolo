import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from acrholo.data import Protein, validate_proteins
from acrholo.training import cross_validation_folds

ABLATION_BRANCHES = {
    "esm": (0, 7680),
    "msa": (7680, 9216),
    "frequency": (9216, 10327),
    "e1": (10327, 12887),
}
ABLATION_METHODS = {
    "full": ("esm", "msa", "frequency", "e1"),
    "without-e1": ("esm", "msa", "frequency"),
    "query-e1": ("esm", "msa", "frequency", "e1"),
    "without-esm": ("msa", "frequency", "e1"),
    "without-msa": ("esm", "frequency", "e1"),
    "without-frequency": ("esm", "msa", "e1"),
    "esm-only": ("esm",),
    "msa-only": ("msa",),
    "frequency-only": ("frequency",),
    "e1-only": ("e1",),
}


def ablation_columns(branches: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.arange(*ABLATION_BRANCHES[branch]) for branch in branches])


def check_ablation() -> None:
    widths = (12887, 10327, 12887, 5207, 11351, 11776, 7680, 1536, 1111, 2560)
    for branches, width in zip(ABLATION_METHODS.values(), widths, strict=True):
        columns = ablation_columns(branches)
        assert len(columns) == width and len(set(columns)) == width
        assert np.all(np.diff(columns) > 0)
    assert np.array_equal(ablation_columns(ABLATION_METHODS["full"]), np.arange(12887))
    assert np.array_equal(ablation_columns(("e1",)), np.arange(10327, 12887))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty analysis table")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _weighted_ablation_metrics(
    weights: np.ndarray, labels: np.ndarray, scores: np.ndarray, decisions: np.ndarray
) -> dict[str, np.ndarray]:
    positive = weights @ labels
    negative = weights @ (1 - labels)
    recall = (weights @ (labels * decisions)) / positive
    specificity = (weights @ ((1 - labels) * (1 - decisions))) / negative
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    pos = np.add.reduceat(weights[:, order] * labels[order], starts, axis=1)
    neg = np.add.reduceat(weights[:, order] * (1 - labels[order]), starts, axis=1)
    auc = (pos * (neg.cumsum(1) - 0.5 * neg)).sum(1) / (positive * negative)
    pos, neg = pos[:, ::-1], neg[:, ::-1]
    cumulative_pos = pos.cumsum(1)
    total = cumulative_pos + neg.cumsum(1)
    precision = np.divide(cumulative_pos, total, out=np.zeros_like(total), where=total > 0)
    return {
        "balanced_accuracy": (recall + specificity) / 2,
        "auprc": (pos * precision).sum(1) / positive,
        "auc": auc,
        "recall": recall,
        "specificity": specificity,
    }


def _ablation_family_weights(
    families: list[str], labels: np.ndarray, rng: np.random.Generator, repeats: int
) -> np.ndarray:
    unique = sorted(set(families))
    lookup = {family: i for i, family in enumerate(unique)}
    codes = np.array([lookup[family] for family in families])
    if any(len(set(labels[codes == i])) != 1 for i in range(len(unique))):
        raise ValueError("Ablation bootstrap requires homogeneous binary families")
    family_labels = np.array([labels[codes == i][0] for i in range(len(unique))])
    counts = np.zeros((repeats, len(unique)))
    for label in (0, 1):
        members = np.flatnonzero(family_labels == label)
        if not len(members):
            raise ValueError("Every held fold must contain both family classes")
        counts[:, members] = rng.multinomial(
            len(members), np.full(len(members), 1 / len(members)), repeats
        )
    return counts[:, codes]


def ablation_uncertainty(output: Path) -> None:
    for name in ("intervals.csv", "paired_intervals.csv"):
        if (output / name).exists():
            raise FileExistsError(f"Analysis table already exists: {output / name}")
    statistics = ("balanced_accuracy", "auprc", "auc", "recall", "specificity")
    operating = {"balanced_accuracy", "recall", "specificity"}
    tables = []
    schemas = {
        "fold_metrics.csv": {
            "method",
            "branches",
            "fold",
            "dimension",
            "policy",
            "policy_valid",
            "threshold",
            "held_n",
            "held_positives",
            *statistics,
        },
        "scores.csv": {
            "method",
            "branches",
            "dimension",
            "fold",
            "id",
            "family",
            "label",
            "score",
            "probability",
            "threshold",
            "policy_valid",
            "decision",
        },
        "summary.csv": {
            "method",
            "branches",
            "dimension",
            "folds",
            "provisional",
            "policy_valid_all_folds",
            *(f"mean_{key}" for key in statistics),
        },
    }
    for name, required in schemas.items():
        with (output / name).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(f"Incomplete ablation schema: {name}")
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError(f"Malformed ablation CSV: {name}")
            tables.append(rows)
    fold_rows, held_rows, summary = tables
    present = {r["method"] for r in fold_rows}
    methods = tuple(
        method for method in ABLATION_METHODS if method != "query-e1" or "query-e1" in present
    )
    expected = {(method, fold) for method in methods for fold in range(5)}
    if (
        len(fold_rows) != len(expected)
        or {(r["method"], int(r["fold"])) for r in fold_rows} != expected
    ):
        raise ValueError("Uncertainty requires all fixed ablations and all five completed folds")
    if len(summary) != len(methods) or {r["method"] for r in summary} != set(methods):
        raise ValueError("Uncertainty requires a completed summary for every ablation")
    if {r["method"] for r in held_rows} != set(methods) or len(
        {(r["method"], r["id"]) for r in held_rows}
    ) != len(held_rows):
        raise ValueError("Held IDs must occur exactly once per ablation")
    if {(r["method"], int(r["fold"])) for r in held_rows} != expected:
        raise ValueError("Held scores must cover all fixed ablations and five folds")
    reference = [r for r in held_rows if r["method"] == "full"]
    family_identity: dict[str, tuple[str, str]] = {}
    for row in reference:
        identity = (row["label"], row["fold"])
        if row["label"] not in {"0", "1"} or not row["id"] or not row["family"]:
            raise ValueError("Held members require IDs, families, and binary labels")
        if family_identity.setdefault(row["family"], identity) != identity:
            raise ValueError("Families must be homogeneous and confined to one held fold")
    metric_rows = {(r["method"], int(r["fold"])): r for r in fold_rows}
    summary_rows = {r["method"]: r for r in summary}
    for method in methods:
        dimension = len(ablation_columns(ABLATION_METHODS[method]))
        branches = "+".join(ABLATION_METHODS[method])
        summary_row = summary_rows[method]
        if (
            int(summary_row["dimension"]) != dimension
            or summary_row["branches"] != branches
            or summary_row["folds"] != "5"
            or summary_row["provisional"] != "False"
            or summary_row["policy_valid_all_folds"] not in {"True", "False"}
        ):
            raise ValueError(f"Invalid completed summary for {method}")
        for fold in range(5):
            row = metric_rows[method, fold]
            if (
                int(row["dimension"]) != dimension
                or row["branches"] != branches
                or row["policy"] != "recall95-continuous"
                or row["policy_valid"] not in {"True", "False"}
            ):
                raise ValueError(f"Invalid policy or configuration for {method} fold {fold}")
        all_valid = all(metric_rows[method, fold]["policy_valid"] == "True" for fold in range(5))
        if (summary_row["policy_valid_all_folds"] == "True") != all_valid:
            raise ValueError(f"Summary policy validity mismatch for {method}")
    observed: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    bootstrap: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    invalid: dict[str, list[int]] = {method: [] for method in methods}
    rng = np.random.default_rng(42)
    for fold in range(5):
        baseline = sorted((r for r in reference if int(r["fold"]) == fold), key=lambda r: r["id"])
        labels = np.array([int(r["label"]) for r in baseline])
        families = [r["family"] for r in baseline]
        weights = _ablation_family_weights(families, labels, rng, 10000)
        identity = [(r["id"], r["label"], r["family"]) for r in baseline]
        for method in methods:
            current = sorted(
                (r for r in held_rows if r["method"] == method and int(r["fold"]) == fold),
                key=lambda r: r["id"],
            )
            if [(r["id"], r["label"], r["family"]) for r in current] != identity:
                raise ValueError(f"Mismatched held members for {method} fold {fold}")
            row = metric_rows[method, fold]
            if (
                len(current) != int(row["held_n"])
                or int(labels.sum()) != int(row["held_positives"])
                or any(
                    r["policy_valid"] != row["policy_valid"]
                    or r["branches"] != row["branches"]
                    or int(r["dimension"]) != int(row["dimension"])
                    for r in current
                )
            ):
                raise ValueError(f"Held metadata mismatch for {method} fold {fold}")
            scores, probabilities, thresholds, decisions = (
                np.array([float(r[key]) for r in current])
                for key in ("score", "probability", "threshold", "decision")
            )
            if not np.isfinite(scores).all() or not (
                np.isfinite(probabilities).all()
                and ((probabilities >= 0) & (probabilities <= 1)).all()
            ):
                raise ValueError(
                    f"Nonfinite scores or invalid probabilities for {method} fold {fold}"
                )
            valid = row["policy_valid"] == "True"
            if valid:
                if not np.isfinite(thresholds).all() or not np.all(
                    thresholds == float(row["threshold"])
                ):
                    raise ValueError(f"Threshold mismatch for {method} fold {fold}")
                if not 0.05 <= float(row["threshold"]) <= 0.95 or not np.array_equal(
                    decisions, probabilities >= thresholds
                ):
                    raise ValueError(f"Invalid decisions for {method} fold {fold}")
            else:
                invalid[method].append(fold)
                if (
                    not np.isnan(thresholds).all()
                    or not np.isnan(decisions).all()
                    or not np.isnan(float(row["threshold"]))
                ):
                    raise ValueError(
                        f"Invalid policy must have NaN cuts and decisions: {method} fold {fold}"
                    )
            observed[method, fold] = _weighted_ablation_metrics(
                np.ones((1, len(labels))), labels, scores, decisions
            )
            bootstrap[method, fold] = _weighted_ablation_metrics(weights, labels, scores, decisions)
            for statistic in statistics:
                np.testing.assert_allclose(
                    observed[method, fold][statistic][0],
                    float(row[statistic]),
                    atol=1e-12,
                    rtol=0,
                    equal_nan=True,
                )
    intervals, paired = [], []
    for method in methods:
        for statistic in statistics:
            point = float(np.mean([observed[method, fold][statistic][0] for fold in range(5)]))
            np.testing.assert_allclose(
                point,
                float(summary_rows[method][f"mean_{statistic}"]),
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            )
            draws = np.mean([bootstrap[method, fold][statistic] for fold in range(5)], axis=0)
            valid_draws = int(np.isfinite(draws).sum())
            low, high = (
                np.quantile(draws, [0.025, 0.975]) if valid_draws == 10000 else (np.nan, np.nan)
            )
            intervals.append(
                dict(
                    method=method,
                    metric=statistic,
                    estimate=point,
                    lower95=float(low),
                    upper95=float(high),
                    valid_draws=valid_draws,
                    policy_valid_all_folds=not invalid[method],
                    invalid_folds=",".join(map(str, invalid[method])),
                )
            )
            if method == "full":
                continue
            fold_deltas = [
                observed["full", fold][statistic][0] - observed[method, fold][statistic][0]
                for fold in range(5)
            ]
            deltas = np.mean(
                [
                    bootstrap["full", fold][statistic] - bootstrap[method, fold][statistic]
                    for fold in range(5)
                ],
                axis=0,
            )
            paired_valid = int(np.isfinite(deltas).sum())
            low, high = (
                np.quantile(deltas, [0.025, 0.975]) if paired_valid == 10000 else (np.nan, np.nan)
            )
            invalid_folds = sorted(set(invalid["full"] + invalid[method]))
            paired.append(
                dict(
                    method="full",
                    control=method,
                    metric=statistic,
                    delta=float(np.mean(fold_deltas)),
                    lower95=float(low),
                    upper95=float(high),
                    fold_wins=sum(delta > 0 for delta in fold_deltas)
                    if statistic not in operating or not invalid_folds
                    else float("nan"),
                    valid_draws=paired_valid,
                    invalid_folds=",".join(map(str, invalid_folds)),
                )
            )
    write_rows(output / "intervals.csv", intervals)
    write_rows(output / "paired_intervals.csv", paired)


POLICY_METRICS = (
    "balanced_accuracy",
    "specificity",
    "recall",
    "positive_precision",
    "f1",
    "mcc",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "calibration_recall",
    "calibration_specificity",
)


def _operating_metrics(rows: list[dict]) -> dict[str, float]:
    from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

    labels = np.array([row["label"] for row in rows])
    decisions = np.array([row["decision"] for row in rows])
    if not np.isfinite(decisions).all():
        return dict.fromkeys(POLICY_METRICS, float("nan"))
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    recall, specificity = float(tp / (tp + fn)), float(tn / (tn + fp))
    return {
        "balanced_accuracy": (recall + specificity) / 2,
        "specificity": specificity,
        "recall": recall,
        "positive_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "f1": float(f1_score(labels, decisions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, decisions)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def ablate(
    values: np.ndarray,
    proteins: list[Protein],
    groups: list[str],
    output: Path,
    *,
    query_e1: np.ndarray | None = None,
    seed: int = 42,
) -> None:
    validate_proteins(proteins, labeled=True)
    if (
        values.shape != (len(proteins), 12887)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
        or len(groups) != len(proteins)
        or any(not isinstance(group, str) or not group.strip() for group in groups)
    ):
        raise ValueError("Ablation requires finite FP64 N x 12887 features and aligned families")
    if query_e1 is not None and (
        query_e1.shape != (len(proteins), 2560)
        or query_e1.dtype != np.float32
        or not np.isfinite(query_e1).all()
    ):
        raise ValueError("Query E1 requires aligned finite FP32 N x 2560 features")
    family_labels: dict[str, int] = {}
    for protein, family in zip(proteins, groups, strict=True):
        if family_labels.setdefault(family, protein.label) != protein.label:
            raise ValueError("Ablation bootstrap requires homogeneous binary families")
    if output.exists():
        raise FileExistsError(f"Ablation requires a fresh output directory: {output}")
    fold_rows, scores, calibrations, summary = [], [], [], []
    for method, branches in ABLATION_METHODS.items():
        if method == "query-e1" and query_e1 is None:
            continue
        matrix = (
            np.concatenate((values[:, :10327], query_e1), axis=1)
            if method == "query-e1" and query_e1 is not None
            else values[:, ablation_columns(branches)]
        )
        method_rows = []
        for metadata, held_rows, calibration_rows in cross_validation_folds(
            matrix, proteins, groups, seed=seed, allow_invalid=True
        ):
            metadata = dict(metadata)
            identity = {"method": method, "branches": "+".join(branches)}
            prediction_metadata = {
                **identity,
                "dimension": metadata["dimension"],
                "calibration_slope": metadata["calibration_slope"],
                "calibration_intercept": metadata["calibration_intercept"],
            }
            held_rows = [{**prediction_metadata, **row} for row in held_rows]
            calibration_rows = [{**prediction_metadata, **row} for row in calibration_rows]
            labels = np.array([row["label"] for row in held_rows])
            raw = np.array([row["score"] for row in held_rows])
            operating = _operating_metrics(held_rows)
            calibration_operating = _operating_metrics(calibration_rows)
            row = {
                **identity,
                **metadata,
                "auprc": float(average_precision_score(labels, raw)),
                "auc": float(roc_auc_score(labels, raw)),
                **operating,
                "calibration_recall": calibration_operating["recall"],
                "calibration_specificity": calibration_operating["specificity"],
            }
            method_rows.append(row)
            scores.extend(held_rows)
            calibrations.extend(calibration_rows)
            print(
                f"{method} fold {metadata['fold']}: AUPRC={row['auprc']:.4f}, "
                f"policy_valid={metadata['policy_valid']}",
                flush=True,
            )
        if len(method_rows) != 5:
            raise ValueError("Ablation requires all five folds for each fixed configuration")
        fold_rows.extend(method_rows)
        summary.append(
            {
                **identity,
                "dimension": matrix.shape[1],
                "folds": 5,
                "policy_valid_all_folds": all(row["policy_valid"] for row in method_rows),
                "provisional": False,
                "mean_auprc": float(np.mean([row["auprc"] for row in method_rows])),
                "mean_auc": float(np.mean([row["auc"] for row in method_rows])),
                "std_auprc": float(np.std([row["auprc"] for row in method_rows], ddof=1)),
                **{
                    f"mean_{key}": float(np.mean([row[key] for row in method_rows]))
                    for key in POLICY_METRICS
                },
            }
        )
    output.mkdir(parents=True)
    for name, rows in (
        ("fold_metrics.csv", fold_rows),
        ("scores.csv", scores),
        ("calibration_scores.csv", calibrations),
        ("summary.csv", summary),
    ):
        write_rows(output / name, rows)
    ablation_uncertainty(output)


def self_check() -> None:
    check_ablation()
    labels = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.2, 0.2, 0.8, -1.0, 0.8])
    families = ["a", "b", "c", "d", "a"]
    decisions = (scores >= 0.3).astype(float)
    shared = _ablation_family_weights(families, labels, np.random.default_rng(42), 100)
    weights = np.concatenate((np.array([[1, 2, 3, 1, 2], [2, 1, 1, 3, 1]]), shared))
    weighted = _weighted_ablation_metrics(weights, labels, scores, decisions)
    for index, weight in enumerate(weights):
        expanded = np.repeat(np.arange(len(labels)), weight.astype(int))
        np.testing.assert_allclose(
            [weighted["auprc"][index], weighted["auc"][index]],
            [
                average_precision_score(labels, scores, sample_weight=weight),
                roc_auc_score(labels, scores, sample_weight=weight),
            ],
            atol=1e-15,
            rtol=0,
        )
        np.testing.assert_allclose(
            [weighted["auprc"][index], weighted["auc"][index]],
            [
                average_precision_score(labels[expanded], scores[expanded]),
                roc_auc_score(labels[expanded], scores[expanded]),
            ],
            atol=1e-15,
            rtol=0,
        )
        expected = _operating_metrics(
            [{"label": labels[i], "decision": decisions[i]} for i in expanded]
        )
        for key in ("balanced_accuracy", "recall", "specificity"):
            np.testing.assert_allclose(weighted[key][index], expected[key], atol=1e-15, rtol=0)
    assert np.array_equal(shared[:, 0], shared[:, 4])
    assert (shared[:, [0, 2]].sum(1) == 2).all() and (shared[:, [1, 3]].sum(1) == 2).all()
    invalid = _weighted_ablation_metrics(weights, labels, scores, np.full(5, np.nan))
    for key in ("balanced_accuracy", "recall", "specificity"):
        assert np.isnan(invalid[key]).all()
    for key in ("auprc", "auc"):
        np.testing.assert_array_equal(invalid[key], weighted[key])
    assert all(
        np.isnan(value) for value in _operating_metrics([{"label": 1, "decision": np.nan}]).values()
    )
