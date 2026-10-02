import csv
import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def reference_exclusions() -> set[str]:
    from .recipe import REFERENCE_EXCLUSIONS_SHA256

    content = files("acrholo").joinpath("reference_exclusions.txt").read_bytes()
    if hashlib.sha256(content).hexdigest() != REFERENCE_EXCLUSIONS_SHA256:
        raise ValueError("The pinned benchmark reference exclusions have changed")
    hashes = set(content.decode("ascii").splitlines())
    if len(hashes) != 1436 or any(
        len(value) != 64 or set(value) - set("0123456789abcdef") for value in hashes
    ):
        raise ValueError("Invalid benchmark sequence exclusions")
    return hashes


@dataclass(frozen=True)
class Protein:
    id: str
    seq: str
    label: int = -1


def validate_proteins(proteins: list[Protein], *, labeled: bool = False) -> None:
    if not proteins or len({p.id for p in proteins}) != len(proteins):
        raise ValueError("Provide nonempty proteins with unique identifiers")
    for protein in proteins:
        if (
            not protein.id.strip()
            or any(char.isspace() for char in protein.id)
            or not protein.seq
            or len(protein.seq) > 4092
            or set(protein.seq) - CANONICAL
            or (labeled and protein.label not in (0, 1))
        ):
            raise ValueError(
                "Require IDs without whitespace, canonical20+X sequences <=4092, "
                "and binary training labels"
            )


def read_proteins(path: Path, *, labeled: bool = False) -> list[Protein]:
    proteins = []
    with path.open() as handle:
        if path.suffix.lower() in {".fa", ".fasta", ".faa", ".fas"}:
            if labeled:
                raise ValueError("Labeled training input requires CSV with label column")
            identifier, residues = None, []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if identifier is not None:
                        proteins.append(Protein(identifier, "".join(residues).upper()))
                    header = line[1:].split()
                    if not header:
                        raise ValueError("FASTA requires a nonempty identifier")
                    identifier, residues = header[0], []
                elif identifier is None:
                    raise ValueError("FASTA sequence must follow a header")
                else:
                    residues.append("".join(line.split()))
            if identifier is not None:
                proteins.append(Protein(identifier, "".join(residues).upper()))
        else:
            reader = csv.DictReader(handle)
            required = {"seq", "label"} if labeled else {"seq"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(f"CSV requires columns: {sorted(required)}")
            for index, row in enumerate(reader):
                if None in row:
                    raise ValueError("CSV row has extra fields")
                label = row.get("label") if labeled else None
                if labeled and label not in {"0", "1"}:
                    raise ValueError("Training labels must be 0 or 1")
                identifier = row.get("id") if "id" in reader.fieldnames else f"query_{index:04d}"
                proteins.append(
                    Protein(
                        identifier or "",
                        "".join((row.get("seq") or "").split()).upper(),
                        int(label) if label is not None else -1,
                    )
                )
    validate_proteins(proteins, labeled=labeled)
    return proteins


def read_families(path: Path, proteins: list[Protein]) -> list[str]:
    validate_proteins(proteins, labeled=True)
    assignments = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"id", "family"} <= set(reader.fieldnames):
            raise ValueError("Family CSV requires id and family columns")
        for row in reader:
            identifier, family = row.get("id"), row.get("family")
            if (
                None in row
                or not identifier
                or not family
                or not family.strip()
                or identifier in assignments
            ):
                raise ValueError("Family CSV requires unique IDs and nonempty families")
            assignments[identifier] = family
    if set(assignments) != {protein.id for protein in proteins}:
        raise ValueError("Family CSV must assign exactly the provided training IDs")
    return [assignments[protein.id] for protein in proteins]


def cluster_training(
    proteins: list[Protein], directory: Path, *, binary: str = "mmseqs", threads: int = 8
) -> list[str]:
    validate_proteins(proteins, labeled=True)
    if threads < 1:
        raise ValueError("Threads must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="train-", dir=directory.resolve()))
    fasta = run / "train.fasta"
    fasta.write_text("".join(f">{p.id}\n{p.seq}\n" for p in proteins))
    subprocess.run(
        [
            binary,
            "easy-cluster",
            str(fasta),
            str(run / "families"),
            str(run / "tmp"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--cluster-mode",
            "1",
            "--single-step-clustering",
            "1",
            "--max-seqs",
            str(len(proteins)),
            "--max-iterations",
            str(len(proteins)),
            "-s",
            "7.5",
            "--threads",
            str(threads),
            "-v",
            "1",
        ],
        check=True,
    )
    assignments = {}
    with (run / "families_cluster.tsv").open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2 or row[1] in assignments:
                raise ValueError("Invalid or duplicate MMseqs family assignment")
            representative, member = row
            assignments[member] = representative
    if set(assignments) != {p.id for p in proteins} or set(assignments.values()) - set(assignments):
        raise ValueError("MMseqs must assign every training protein to a training representative")
    return [assignments[p.id] for p in proteins]


def validation_folds(
    proteins: list[Protein], groups: list[str], n_splits: int = 5, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    from sklearn.model_selection import StratifiedGroupKFold

    validate_proteins(proteins, labeled=True)
    if (
        n_splits < 2
        or len(proteins) != len(groups)
        or len(set(groups)) < n_splits
        or any(not group.strip() for group in groups)
    ):
        raise ValueError("Provide one nonempty family per protein and at least n_splits families")
    labels = [p.label for p in proteins]
    if set(labels) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        (fit.tolist(), held.tolist())
        for fit, held in splitter.split(proteins, labels, groups=groups)
    ]
    for fit, held in folds:
        if (
            {groups[i] for i in fit} & {groups[i] for i in held}
            or {labels[i] for i in fit} != {0, 1}
            or {labels[i] for i in held} != {0, 1}
        ):
            raise ValueError(
                "Family folds require disjoint families and both classes in each partition"
            )
    return folds


def split_training(
    proteins: list[Protein], groups: list[str], seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    folds = [held for _, held in validation_folds(proteins, groups, seed=seed)]
    return sorted(folds[0] + folds[1] + folds[2]), sorted(folds[3]), sorted(folds[4])


def self_check() -> None:
    proteins = [Protein(f"train_{index:04d}", "AX", index % 2) for index in range(100)]
    groups = [protein.id for protein in proteins]
    fit, selection, reserve = split_training(proteins, groups)
    assert sorted(fit + selection + reserve) == list(range(100))
    assert not set(fit) & set(reserve) and not set(selection) & set(reserve)
    for invalid in ([Protein("", "A")], [Protein("query", "B")], proteins + proteins[:1]):
        try:
            validate_proteins(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid proteins accepted")
