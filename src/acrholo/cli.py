import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data import Protein, read_proteins, validate_proteins
from .model import FEATURE_DIMENSION, feature_matrix, load_checkpoint, predict
from .recipe import feature_recipe


def save_features(
    path: Path, proteins: list[Protein], values: np.ndarray, *, query_e1: bool = False
) -> None:
    validate_proteins(proteins)
    dimension, dtype = (2560, np.float32) if query_e1 else (FEATURE_DIMENSION, np.float64)
    if (
        values.shape != (len(proteins), dimension)
        or values.dtype != dtype
        or not np.isfinite(values).all()
    ):
        raise ValueError("Feature rows must match the representation's dimension and precision")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray([protein.id for protein in proteins]),
            sequences=np.asarray([protein.seq for protein in proteins]),
            values=values,
            recipe=json.dumps(feature_recipe(query_e1=query_e1), sort_keys=True),
        )


def load_features(path: Path, *, query_e1: bool = False) -> tuple[list[Protein], np.ndarray]:
    dimension, dtype = (2560, np.float32) if query_e1 else (FEATURE_DIMENSION, np.float64)
    with np.load(path, allow_pickle=False) as saved:
        identifiers, sequences, values, recipe = (
            saved[key] for key in ("ids", "sequences", "values", "recipe")
        )
        if (
            identifiers.ndim != 1
            or identifiers.dtype.kind != "U"
            or sequences.shape != identifiers.shape
            or sequences.dtype.kind != "U"
            or values.shape != (len(identifiers), dimension)
            or values.dtype != dtype
            or not np.isfinite(values).all()
            or recipe.shape != ()
            or recipe.dtype.kind != "U"
            or json.loads(str(recipe)) != feature_recipe(query_e1=query_e1)
        ):
            raise ValueError("Invalid feature archive or incompatible AcrHolo recipe")
        proteins = [
            Protein(str(identifier), str(sequence))
            for identifier, sequence in zip(identifiers, sequences, strict=True)
        ]
        validate_proteins(proteins)
        return proteins, values.copy()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_features(arguments: argparse.Namespace) -> None:
    from .features import prepare_features
    from .msa import prepare_msa
    from .profiles import frequency_cache_directory, prepare_frequency

    proteins = read_proteins(arguments.input)
    profiles, available = prepare_frequency(
        proteins,
        arguments.database,
        arguments.cache,
        binary=arguments.mmseqs,
        cache_only=arguments.cache_only,
    )
    embeddings = prepare_features(
        proteins,
        arguments.cache,
        arguments.device,
        arguments.batch_size,
        cache_only=arguments.cache_only,
    )
    msa = prepare_msa(
        proteins,
        arguments.database,
        arguments.msa_checkpoint,
        arguments.cache,
        arguments.device,
        binary=arguments.mmseqs,
        cache_only=arguments.cache_only,
    )
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    alignment_directory = frequency_cache_directory(
        arguments.database, arguments.cache, binary=arguments.mmseqs
    )
    context = extract_e1_features(arguments, proteins, alignment_directory)
    values = feature_matrix(embeddings, msa, profiles, available, context)
    save_features(arguments.output, proteins, values)


def extract_e1_features(
    arguments: argparse.Namespace, proteins: list[Protein], alignment_directory: Path | None = None
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="acrholo-e1-") as temporary:
        directory = Path(temporary)
        queries, output = directory / "queries.csv", directory / "context.npy"
        write_rows(queries, [{"id": protein.id, "seq": protein.seq} for protein in proteins])
        command = [
            # Resolving a venv interpreter symlink would discard its environment.
            str(arguments.e1_python.absolute()),
            "-m",
            "acrholo.e1",
            "--input",
            str(queries),
            "--cache",
            str(arguments.cache.resolve()),
            "--source",
            str(arguments.e1_source.resolve()),
            "--weights",
            str(arguments.e1_weights.resolve()),
            "--output",
            str(output),
        ]
        if alignment_directory is not None:
            command.extend(("--alignments", str(alignment_directory.resolve())))
        if arguments.cache_only:
            command.append("--cache-only")
        environment = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "FAST_BLOCK_MASK": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        }
        subprocess.run(command, env=environment, check=True)
        return np.load(output, allow_pickle=False)


def training_inputs(arguments: argparse.Namespace) -> tuple[np.ndarray, list[Protein], list[str]]:
    from .data import cluster_training, read_families

    proteins = read_proteins(arguments.input, labeled=True)
    cached, values = load_features(arguments.features)
    if [(p.id, p.seq) for p in proteins] != [(p.id, p.seq) for p in cached]:
        raise ValueError(
            "Training CSV and feature archive must have identical IDs, sequences and order"
        )
    groups = (
        read_families(arguments.families, proteins)
        if arguments.families
        else cluster_training(
            proteins,
            arguments.cache / "families",
            binary=arguments.mmseqs,
            threads=arguments.threads,
        )
    )
    return values, proteins, groups


def check() -> None:
    from . import ablation, data, e1, features, model, msa, profiles, training

    for module in (data, e1, features, model, msa, profiles, training, ablation):
        module.self_check()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        proteins = [Protein("synthetic", "AX")]
        values = np.zeros((1, FEATURE_DIMENSION), dtype=np.float64)
        path = directory / "features.npz"
        save_features(path, proteins, values)
        observed, restored = load_features(path)
        assert observed == proteins and np.array_equal(values, restored)
        query_path = directory / "query-e1.npz"
        query_values = np.zeros((1, 2560), dtype=np.float32)
        save_features(query_path, proteins, query_values, query_e1=True)
        observed, restored = load_features(query_path, query_e1=True)
        assert observed == proteins and np.array_equal(query_values, restored)
        try:
            load_features(query_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Query-only E1 accepted as full AcrHolo features")
        try:
            save_features(path, proteins, values)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing feature output overwritten")
    print("AcrHolo numerical and input checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="AcrHolo anti-CRISPR classification")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Run compact synthetic numerical checks")
    reference = commands.add_parser("reference", help="Build the pinned encoded GPU reference")
    reference.add_argument("--input", type=Path, required=True, help="UniRef50 2026_03 FASTA.gz")
    reference.add_argument(
        "--database", type=Path, required=True, help="Fresh padded database prefix"
    )
    reference.add_argument("--mmseqs", default="mmseqs")
    reference.add_argument("--threads", type=int, default=2)
    extraction = commands.add_parser("features", help="Extract the four frozen feature branches")
    query = commands.add_parser("query-e1", help="Extract query-only E1 for the matched control")
    for command in (extraction, query):
        command.add_argument(
            "--input", type=Path, required=True, help="FASTA or CSV with seq and optional id"
        )
        command.add_argument(
            "--output", type=Path, required=True, help="Fresh feature archive (.npz)"
        )
        command.add_argument("--cache", type=Path, default=Path("cache"))
        command.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads")
        command.add_argument("--e1-python", type=Path, required=True)
        command.add_argument("--e1-source", type=Path, required=True)
        command.add_argument("--e1-weights", type=Path, required=True)
        command.add_argument("--cache-only", action="store_true")
    extraction.add_argument("--database", type=Path, required=True)
    extraction.add_argument("--msa-checkpoint", type=Path, required=True)
    extraction.add_argument("--mmseqs", default="mmseqs")
    extraction.add_argument("--device", default="cuda")
    extraction.add_argument("--batch-size", type=int, default=8)
    prediction = commands.add_parser("predict", help="Score a feature archive on CPU")
    prediction.add_argument("--features", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True, help="Fresh prediction CSV")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--threads", type=int, default=8)
    for command in ("train", "cross-validate", "ablate"):
        training = commands.add_parser(
            command, help="Use only the explicitly supplied training set"
        )
        training.add_argument(
            "--input", type=Path, required=True, help="CSV with seq, label and optional id"
        )
        training.add_argument("--features", type=Path, required=True)
        training.add_argument("--output", type=Path, required=True)
        training.add_argument("--families", type=Path, help="CSV with id,family")
        training.add_argument("--cache", type=Path, default=Path("cache"))
        training.add_argument("--mmseqs", default="mmseqs")
        training.add_argument("--threads", type=int, default=8)
        training.add_argument("--seed", type=int, default=42)
        if command == "ablate":
            training.add_argument(
                "--query-features", type=Path, help="Query-only E1 archive from query-e1"
            )
    arguments = parser.parse_args()
    if getattr(arguments, "threads", 1) < 1:
        parser.error("Threads must be positive")
    if getattr(arguments, "output", None) is not None and arguments.output.exists():
        raise FileExistsError(arguments.output)
    torch.set_num_threads(getattr(arguments, "threads", 8))
    torch.set_float32_matmul_precision("high")
    if arguments.command == "check":
        check()
    elif arguments.command == "reference":
        from .profiles import prepare_reference

        prepare_reference(
            arguments.input, arguments.database, binary=arguments.mmseqs, threads=arguments.threads
        )
    elif arguments.command == "features":
        extract_features(arguments)
    elif arguments.command == "query-e1":
        proteins = read_proteins(arguments.input)
        save_features(
            arguments.output,
            proteins,
            extract_e1_features(arguments, proteins),
            query_e1=True,
        )
    elif arguments.command == "predict":
        proteins, values = load_features(arguments.features)
        outputs = predict(values, load_checkpoint(arguments.checkpoint))
        write_rows(
            arguments.output,
            [
                {
                    "id": protein.id,
                    **{name: values[index].item() for name, values in outputs.items()},
                }
                for index, protein in enumerate(proteins)
            ],
        )
    elif arguments.command == "train":
        from .training import fit

        checkpoint = fit(*training_inputs(arguments), seed=arguments.seed)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("xb") as handle:
            torch.save(checkpoint, handle)
    elif arguments.command == "ablate":
        from .ablation import ablate

        values, proteins, groups = training_inputs(arguments)
        context = None
        if arguments.query_features:
            queries, context = load_features(arguments.query_features, query_e1=True)
            if [(p.id, p.seq) for p in proteins] != [(p.id, p.seq) for p in queries]:
                raise ValueError("Query-only E1 must match training IDs, sequences and order")
        ablate(values, proteins, groups, arguments.output, query_e1=context, seed=arguments.seed)
    else:
        from .training import cross_validate

        write_rows(
            arguments.output, cross_validate(*training_inputs(arguments), seed=arguments.seed)
        )
