import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .data import Protein, reference_exclusions
from .recipe import MMSEQS_VERSION as GPU_VERSION
from .recipe import REFERENCE_SHA256, UNIREF_BYTES, UNIREF_MD5

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_GROUPS = [
    [0, 4],
    [3, 6],
    [13, 16],
    [7, 8],
    [9, 10],
    [11, 12],
    [14, 15],
    [1, 2],
    [5, 17],
    [18, 19],
    [0, 1, 2, 3],
]
FREQUENCY_SETTINGS = (
    "evalue=0.001|query_coverage=0.5|cov_mode=2|sensitivity=7.5|max_seqs=1000|"
    "sort=evalue,target|deduplicate=target_sequence|exclude=self|top=16|"
    "weights=uniform|pseudocount=1|query_noncanonical=missing|descriptors=1"
)


def pooled_profile(probabilities: np.ndarray, seq: str) -> np.ndarray:
    if (
        probabilities.shape != (len(seq), 20)
        or not seq
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Profile must contain twenty finite values in [0, 1] per query residue")
    probabilities = probabilities.astype(np.float32, copy=False)
    reduced = np.stack([probabilities[:, group].mean(axis=1) for group in AA_GROUPS], axis=1)
    covariance = []
    for matrix in (reduced, probabilities):
        centered = matrix - matrix.mean(axis=0)
        covariance.extend(
            float(np.mean(column[:-lag] * column[lag:])) if len(seq) > lag else 0.0
            for column in centered.T
            for lag in range(1, 11)
        )
    composition = np.zeros((20, 20), dtype=np.float32)
    counts = np.zeros(20, dtype=np.float32)
    for residue, row in zip(seq, probabilities, strict=True):
        index = AA_ORDER.find(residue)
        if index >= 0:
            composition[index] += row
            counts[index] += 1
    composition /= np.maximum(counts[:, None], 1)
    coupling = (
        probabilities[:-1].T @ probabilities[1:] / float(len(seq) - 1)
        if len(seq) > 1
        else np.zeros((20, 20), dtype=np.float32)
    )
    features = np.concatenate([covariance, composition.ravel(), coupling.ravel()]).astype(
        np.float32
    )
    assert features.shape == (1110,) and np.isfinite(features).all()
    return features


GPU_NORMALIZATION = "uppercase_qaln_taln_tseq"
GPU_ENCODING = "canonical20_unchanged;B->D;Z->E;U->X;O->X;J->L;X->X"


def _encode_reference(source: Path, destination: Path, exclusions: set[str]) -> tuple[int, int]:
    retained = removed = 0
    identifier = ""
    parts = []
    with (
        gzip.open(source, "rt", encoding="ascii") as reader,
        destination.open("w", encoding="ascii") as writer,
    ):

        def emit() -> bool:
            sequence = "".join(parts).upper()
            if not identifier or not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZJUO"):
                raise ValueError("Reference FASTA requires complete valid protein records")
            if hashlib.sha256(sequence.encode()).hexdigest() in exclusions:
                return False
            writer.write(f">{identifier}\n{sequence}\n")
            return True

        for line in reader:
            content = line.rstrip("\r\n")
            if content.startswith(">"):
                if identifier:
                    kept = emit()
                    retained += int(kept)
                    removed += int(not kept)
                tokens = content[1:].split()
                if not tokens:
                    raise ValueError("Reference FASTA requires a nonempty header")
                identifier, parts = tokens[0], []
            elif content:
                if not identifier:
                    raise ValueError("Reference FASTA residues precede the first header")
                parts.append(content)
        kept = emit()
        retained += int(kept)
        removed += int(not kept)
    return retained, removed


@lru_cache(maxsize=4)
def _verify_reference_files(files: tuple[tuple[str, int, int], ...]) -> None:
    for (filename, _, _), suffix in zip(
        files, ("", ".index", ".dbtype", "_h", "_h.index", "_h.dbtype"), strict=True
    ):
        with Path(filename).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != REFERENCE_SHA256[suffix]:
            raise ValueError(
                f"Reference database content differs from the pinned target: {filename}"
            )


def prepare_reference(
    source: Path, database: Path, *, binary: str = "mmseqs", threads: int = 2
) -> Path:
    if threads != 2:
        raise ValueError("The pinned reference preparation recipe requires two threads")
    executable = shutil.which(binary)
    if executable is None:
        raise FileNotFoundError(binary)
    if subprocess.check_output([executable, "version"], text=True).strip() != GPU_VERSION:
        raise ValueError("Reference preparation requires the pinned GPU MMseqs version")
    if source.stat().st_size != UNIREF_BYTES:
        raise ValueError("Reference gzip size differs from pinned UniRef50")
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "md5").hexdigest() != UNIREF_MD5:
            raise ValueError("Reference gzip checksum differs from pinned UniRef50")
    database = database.resolve()
    if any(database.parent.glob(f"{database.name}*")):
        raise FileExistsError(f"Reference destination prefix already exists: {database}")
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reference-", dir=database.parent) as temporary:
        scratch = Path(temporary)
        encoded, unpadded = scratch / "database.fasta", scratch / "unpadded"
        retained, removed = _encode_reference(source, encoded, reference_exclusions())
        if (retained, removed) != (38839206, 821):
            raise ValueError("Reference filtering counts differ from the pinned screened target")
        subprocess.run(
            [
                executable,
                "createdb",
                str(encoded),
                str(unpadded),
                "--threads",
                str(threads),
                "--shuffle",
                "0",
                "--createdb-mode",
                "0",
            ],
            check=True,
        )
        subprocess.run(
            [
                executable,
                "makepaddedseqdb",
                str(unpadded),
                str(database),
                "--threads",
                str(threads),
            ],
            check=True,
        )
    gpu_retrieval_identity(database, executable)
    return database


def gpu_retrieval_identity(database: Path, binary: str) -> dict:
    executable = shutil.which(binary)
    if executable is None:
        raise FileNotFoundError(binary)
    version = subprocess.check_output([executable, "version"], text=True).strip()
    if version != GPU_VERSION:
        raise ValueError("GPU retrieval requires the screened CUDA13 MMseqs version")
    with Path(executable).open("rb") as handle:
        binary_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    files = [
        Path(f"{database.resolve()}{suffix}")
        for suffix in ("", ".index", ".dbtype", "_h", "_h.index", "_h.dbtype")
    ]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError(f"Incomplete padded GPU target at {database}")
    fingerprint = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files)
    _verify_reference_files(fingerprint)
    return {
        "backend": "mmseqs_gpu",
        "binary_version": version,
        "binary_sha256": binary_sha,
        "padded_files": list(fingerprint),
        "search": (
            "gpu=1;threads=2;evalue=0.001;qcov=0.5;cov-mode=2;max-seqs=1000;"
            "alignment=1;defaults-pinned-to-binary"
        ),
        "format": "query,target,evalue,fident,qstart,qend,qlen,qcov,tcov,qaln,taln,tseq",
        "alignment_normalization": GPU_NORMALIZATION,
        "target_alphabet_encoding": GPU_ENCODING,
    }


def frequency_cache_directory(database: Path, cache: Path, *, binary: str = "mmseqs") -> Path:
    identity = gpu_retrieval_identity(database, binary)
    identity["frequency_recipe"] = FREQUENCY_SETTINGS.replace("|sensitivity=7.5", "")
    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _frequency_profile(seq: str, homologs: np.ndarray) -> np.ndarray:
    query = np.asarray([AA_ORDER.find(residue) for residue in seq])
    counts = np.zeros((len(seq), 20), dtype=np.float32)
    valid_query = query >= 0
    counts[np.flatnonzero(valid_query), query[valid_query]] = 1
    counts += np.float32(0.05)
    valid = homologs >= 0
    homolog_counts = np.zeros((len(seq), 20), dtype=np.int64)
    rows, positions = np.nonzero(valid)
    np.add.at(homolog_counts, (positions, homologs[rows, positions]), 1)
    counts += homolog_counts
    denominator = (1 + valid_query + valid.sum(axis=0)).astype(np.float32)
    return counts / denominator[:, None]


def _frequency_from_hits(seq: str, hits: list[list[str]]) -> tuple[np.ndarray, bool]:
    selected, seen = [], set()
    for row in sorted(hits, key=lambda row: (float(row[2]), row[1])):
        target = row[11]
        if target == seq or target in seen:
            continue
        seen.add(target)
        selected.append(row)
        if len(selected) == 16:
            break
    if not selected:
        return np.zeros(1110, dtype=np.float32), False
    homologs = np.full((len(selected), len(seq)), -1, dtype=np.int64)
    for index, row in enumerate(selected):
        start, end, length = map(int, row[4:7])
        query_alignment, target_alignment = row[9:11]
        if (
            length != len(seq)
            or not 1 <= start <= end <= len(seq)
            or len(query_alignment) != len(target_alignment)
        ):
            raise ValueError("Malformed MMseqs query alignment")
        position = start - 1
        for query, target in zip(query_alignment, target_alignment, strict=True):
            if query == "-":
                continue
            if position >= end or (
                query != seq[position] and (query in AA_ORDER or seq[position] in AA_ORDER)
            ):
                raise ValueError("MMseqs alignment does not match the requested query")
            homologs[index, position] = AA_ORDER.find(target)
            position += 1
        if position != end:
            raise ValueError("MMseqs alignment length does not match query coordinates")
    return pooled_profile(_frequency_profile(seq, homologs), seq), True


def _validate_alignments(seq: str, hits: list[list[str]]) -> None:
    if not seq or any(c < "A" or c > "Z" for c in seq):
        raise ValueError("Queries must be nonempty uppercase amino-acid sequences")
    key = hashlib.sha256(seq.encode()).hexdigest()
    for row in hits:
        if len(row) != 12 or row[0] != key or not row[1] or not row[11]:
            raise ValueError("Malformed MMseqs alignment row or query hash")
        if any(any(c.isspace() for c in value) for value in row):
            raise ValueError("Unexpected whitespace in MMseqs alignment fields")
        evalue, identity, qcov, tcov = map(float, (row[2], row[3], row[7], row[8]))
        start, end, length = map(int, row[4:7])
        aligned = row[9].replace("-", "")
        if (
            not np.isfinite([evalue, identity, qcov, tcov]).all()
            or evalue < 0
            or not all(0 <= value <= 1 for value in (identity, qcov, tcov))
            or length != len(seq)
            or not 1 <= start <= end <= length
            or len(row[9]) != len(row[10])
            or len(aligned) != end - start + 1
            or any(
                observed != expected and (observed in AA_ORDER or expected in AA_ORDER)
                for observed, expected in zip(aligned, seq[start - 1 : end], strict=True)
            )
        ):
            raise ValueError("MMseqs alignment does not match its query or coordinates")


def save_alignments(directory: Path, seq: str, hits: list[list[str]]) -> None:
    _validate_alignments(seq, hits)
    key = hashlib.sha256(seq.encode()).hexdigest()
    rows = np.asarray(hits, dtype=str).reshape((-1, 12))
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as handle:
        np.savez_compressed(handle, seq=seq, hits=rows)
        temporary = Path(handle.name)
    temporary.replace(directory / f"{key}.hits.npz")


def load_alignments(directory: Path, seq: str) -> list[list[str]]:
    key = hashlib.sha256(seq.encode()).hexdigest()
    with np.load(directory / f"{key}.hits.npz", allow_pickle=False) as saved:
        rows = saved["hits"]
        if (
            saved["seq"].shape != ()
            or str(saved["seq"]) != seq
            or rows.ndim != 2
            or rows.shape[1] != 12
            or rows.dtype.kind != "U"
        ):
            raise ValueError("Invalid raw-alignment cache")
        hits = rows.tolist()
    _validate_alignments(seq, hits)
    return hits


def prepare_alignments(
    proteins: list[Protein],
    database: Path,
    cache: Path,
    threads: int = 2,
    *,
    binary: str = "mmseqs",
) -> list[list[list[str]]]:
    if threads != 2:
        raise ValueError("The pinned GPU retrieval recipe requires two MMseqs threads")
    if not proteins:
        return []
    if any(not p.seq or any(c < "A" or c > "Z" for c in p.seq) for p in proteins):
        raise ValueError("Queries must be nonempty uppercase amino-acid sequences")
    directory = frequency_cache_directory(database, cache, binary=binary)
    sequences = {hashlib.sha256(p.seq.encode()).hexdigest(): p.seq for p in proteins}
    pending = {
        key: seq for key, seq in sequences.items() if not (directory / f"{key}.hits.npz").exists()
    }
    if pending:
        work = Path(tempfile.mkdtemp(prefix="search-", dir=directory))
        fasta, report = work / "queries.fasta", work / "hits.tsv"
        fasta.write_text("".join(f">{key}\n{seq}\n" for key, seq in pending.items()))
        print(
            f"Searching MMseqs (gpu, threads={threads}) "
            f"for {len(pending)} uncached sequences; "
            f"large references can take hours. Log: {work / 'mmseqs.log'}",
            file=sys.stderr,
            flush=True,
        )
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        with (work / "mmseqs.log").open("w") as log:
            subprocess.run(
                [
                    binary,
                    "easy-search",
                    str(fasta),
                    str(database.resolve()),
                    str(report),
                    str(work / "tmp"),
                    "--format-output",
                    "query,target,evalue,fident,qstart,qend,qlen,qcov,tcov,qaln,taln,tseq",
                    "--threads",
                    str(threads),
                    "-e",
                    "0.001",
                    "-c",
                    "0.5",
                    "--cov-mode",
                    "2",
                    "--gpu",
                    "1",
                    "--max-seqs",
                    "1000",
                    "-a",
                    "1",
                    "--remove-tmp-files",
                    "0",
                    "-v",
                    "2",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        hits: dict[str, list[list[str]]] = {key: [] for key in pending}
        with report.open() as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != 12 or row[0] not in pending:
                    raise ValueError("Malformed MMseqs alignment output")
                for column in (9, 10, 11):
                    row[column] = row[column].upper()
                hits[row[0]].append(row)
        for key, seq in pending.items():
            _validate_alignments(seq, hits[key])
        for key, seq in pending.items():
            save_alignments(directory, seq, hits[key])
    return [load_alignments(directory, protein.seq) for protein in proteins]


def prepare_frequency(
    proteins: list[Protein],
    database: Path,
    cache: Path,
    threads: int = 2,
    *,
    binary: str = "mmseqs",
    cache_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if threads != 2:
        raise ValueError("The pinned GPU retrieval recipe requires two MMseqs threads")
    if not proteins:
        return np.empty((0, 1110), dtype=np.float32), np.empty(0, dtype=bool)
    if any(not p.seq or any(c < "A" or c > "Z" for c in p.seq) for p in proteins):
        raise ValueError("Queries must be nonempty uppercase amino-acid sequences")
    directory = frequency_cache_directory(database, cache, binary=binary)
    missing = {
        hashlib.sha256(p.seq.encode()).hexdigest(): p
        for p in proteins
        if not (directory / f"{hashlib.sha256(p.seq.encode()).hexdigest()}.npz").exists()
    }
    if missing:
        if cache_only:
            raise FileNotFoundError(
                f"Missing {len(missing)} required frequency caches in {directory}"
            )
        hits = prepare_alignments(list(missing.values()), database, cache, threads, binary=binary)
        for (key, protein), rows in zip(missing.items(), hits, strict=True):
            pooled, available = _frequency_from_hits(protein.seq, rows)
            destination = directory / f"{key}.npz"
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as f:
                np.savez_compressed(f, seq=protein.seq, pooled=pooled, available=available)
                temporary = Path(f.name)
            temporary.replace(destination)
    features, availability = [], []
    for protein in proteins:
        key = hashlib.sha256(protein.seq.encode()).hexdigest()
        with np.load(directory / f"{key}.npz", allow_pickle=False) as saved:
            pooled, available = saved["pooled"], saved["available"]
            if (
                str(saved["seq"]) != protein.seq
                or pooled.shape != (1110,)
                or pooled.dtype != np.float32
                or not np.isfinite(pooled).all()
                or available.shape != ()
                or available.dtype != bool
                or (not bool(available) and np.any(pooled != 0))
            ):
                raise ValueError(f"Invalid frequency cache for {protein.id}")
            features.append(pooled.copy())
            availability.append(bool(available))
    return np.stack(features), np.asarray(availability, dtype=bool)


def self_check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        source, destination = Path(temporary) / "reference.gz", Path(temporary) / "encoded.fa"
        with gzip.open(source, "wt", encoding="ascii") as handle:
            handle.write(
                ">excluded description\nab\nj\n>retained description\nAD\nL\n>ambiguous\nbzjuo\n"
            )
        exclusions = {hashlib.sha256(b"ABJ").hexdigest()}
        assert _encode_reference(source, destination, exclusions) == (2, 1)
        assert destination.read_text() == ">retained\nADL\n>ambiguous\nBZJUO\n"
        for invalid in ("ACX\n", ">empty\n", ">first\nA\n>empty\n", ">bad\nA*\n", "> \nA\n"):
            with gzip.open(source, "wt", encoding="ascii") as handle:
                handle.write(invalid)
            try:
                _encode_reference(source, destination, set())
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid reference FASTA accepted")
    sequence = "ABZUOJXCD"
    row = [
        hashlib.sha256(sequence.encode()).hexdigest(),
        "target",
        "0.001",
        "1",
        "1",
        "9",
        "9",
        "1",
        "1",
        sequence,
        "ADEXXLXCD",
        "ADEXXLXCD",
    ]
    _validate_alignments(sequence, [row])
    vector = pooled_profile(np.full((2, 20), 0.5), "AA")
    assert np.array_equal(vector[:310], np.zeros(310))
    assert np.array_equal(vector[310:330], np.full(20, 0.5))
    assert np.array_equal(vector[330:710], np.zeros(380))
    assert np.array_equal(vector[710:], np.full(400, 0.25))
    assert np.array_equal(vector, pooled_profile(np.full((2, 20), 0.5), "AA"))
    probabilities = _frequency_profile("AX", np.asarray([[0, -1]]))
    assert np.isclose(probabilities[0, 0], 2.05 / 3)
    assert np.allclose(probabilities[1], 0.05)
    alignment = ["q", "target", "0.001", "0.5", "1", "2", "2", "1", "1", "AX", "CA", "CA"]
    assert _frequency_from_hits("AX", [alignment])[1]
    for seq, start, query in (("AR", "1", "AX"), ("AX", "0", "AX"), ("AX", "1", "AA")):
        invalid = alignment.copy()
        invalid[4], invalid[9] = start, query
        try:
            _frequency_from_hits(seq, [invalid])
        except ValueError:
            continue
        raise AssertionError("Invalid query alignment was accepted")
