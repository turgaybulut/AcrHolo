import argparse
import hashlib
import tempfile
from importlib.metadata import version
from pathlib import Path

import esm
import numpy as np
import torch

from .data import Protein
from .profiles import frequency_cache_directory, prepare_alignments
from .recipe import MSA_MODEL as MODEL_NAME
from .recipe import MSA_REGRESSION_SHA256, MSA_STATE_SHA256

RECIPE = (
    "diverse64-query-first-full-hamming-tie-source-order-X-v1|fp32-highest-batch1|layer12|"
    "window1023-stride511-end|single-window-strided-pool|residue-overlap-mean|mean-popstd"
)


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _regression_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{MODEL_NAME}-contact-regression.pt")


def msa_cache_directory(
    database: Path, checkpoint: Path, cache: Path, *, binary: str = "mmseqs"
) -> Path:
    reference = frequency_cache_directory(database, cache, binary=binary).name
    identity = (
        reference,
        _digest(checkpoint),
        _digest(_regression_path(checkpoint)),
        version("fair-esm"),
        RECIPE,
    )
    directory = cache / hashlib.sha256(repr(identity).encode()).hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def diverse_msa(identifier: str, sequence: str, hits: list[list[str]]) -> list[tuple[str, str]]:
    rows = [(identifier, sequence)]
    seen = set()
    for hit in sorted(hits, key=lambda row: (float(row[2]), row[1])):
        target = hit[11]
        if target == sequence or target in seen:
            continue
        seen.add(target)
        start, end, length = map(int, hit[4:7])
        qaln, taln = hit[9:11]
        aligned_query = qaln.replace("-", "")
        if (
            not 1 <= start <= end <= len(sequence)
            or length != len(sequence)
            or len(qaln) != len(taln)
            or len(aligned_query) != end - start + 1
            or any(
                observed != expected
                and (observed in "ACDEFGHIKLMNPQRSTVWY" or expected in "ACDEFGHIKLMNPQRSTVWY")
                for observed, expected in zip(aligned_query, sequence[start - 1 : end], strict=True)
            )
        ):
            raise ValueError("Invalid query-coordinate alignment")
        aligned = ["-"] * len(sequence)
        position = start - 1
        for query, residue in zip(qaln, taln, strict=True):
            if query != "-":
                aligned[position] = residue if residue in "ACDEFGHIKLMNPQRSTVWYXBZUO-" else "X"
                position += 1
        assert position == end
        rows.append((hit[1], "".join(aligned)))
    if len(rows) <= 64:
        return rows
    array = np.array([list(seq.encode()) for _, seq in rows], dtype=np.uint8)
    chosen = [0]
    distance = np.zeros(len(rows), dtype=np.int64)
    for _ in range(63):
        distance += np.count_nonzero(array != array[chosen[-1]], axis=1)
        candidate_scores = distance.copy()
        candidate_scores[chosen] = -1
        chosen.append(int(np.argmax(candidate_scores)))
    return [rows[index] for index in sorted(chosen)]


def _window_starts(length: int) -> list[int]:
    if length <= 1023:
        return [0]
    return sorted(set([*range(0, length - 1023 + 1, 511), length - 1023]))


@torch.inference_mode()
def _pooled_msa(model, converter, rows: list[tuple[str, str]], device: str) -> np.ndarray:
    length = len(rows[0][1])
    residues = torch.zeros((length, 768), dtype=torch.float32, device=device)
    counts = torch.zeros((length, 1), dtype=torch.float32, device=device)
    for start in _window_starts(length):
        stop = min(start + 1023, length)
        window = [(name, seq[start:stop].replace("J", "X")) for name, seq in rows]
        _, _, tokens = converter(window)
        assert tokens.shape == (1, len(rows), stop - start + 1)
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            output = model(tokens.to(device), repr_layers=[12], return_contacts=False)
        embedding = output["representations"][12][0, 0, 1 : stop - start + 1]
        assert embedding.shape == (stop - start, 768)
        if length <= 1023:
            pooled = torch.cat((embedding.mean(0), embedding.std(0, correction=0))).cpu().numpy()
            if not np.isfinite(pooled).all():
                raise ValueError("Invalid MSA representation")
            return pooled
        residues[start:stop] += embedding
        counts[start:stop] += 1
        del output, embedding, tokens
    residues /= counts
    pooled = torch.cat((residues.mean(0), residues.std(0, correction=0))).cpu().numpy()
    if pooled.shape != (1536,) or not np.isfinite(pooled).all():
        raise ValueError("Invalid MSA representation")
    return pooled


def prepare_msa(
    proteins: list[Protein],
    database: Path,
    checkpoint: Path,
    cache: Path,
    device: str = "cuda",
    *,
    binary: str = "mmseqs",
    cache_only: bool = False,
) -> np.ndarray:
    if not proteins:
        return np.empty((0, 1536), dtype=np.float32)
    if any(
        not p.seq or any(c not in "ACDEFGHIKLMNPQRSTVWYXBZJUO" for c in p.seq) for p in proteins
    ):
        raise ValueError("MSA queries must be nonempty uppercase amino-acid sequences")
    directory = msa_cache_directory(database, checkpoint, cache, binary=binary)
    sequences = {hashlib.sha256(p.seq.encode()).hexdigest(): p for p in proteins}
    pending = {key: p for key, p in sequences.items() if not (directory / f"{key}.npz").exists()}
    if pending:
        if cache_only:
            raise FileNotFoundError(f"Missing {len(pending)} required MSA caches in {directory}")
        alignments = prepare_alignments(list(pending.values()), database, cache, binary=binary)
        with torch.serialization.safe_globals([argparse.Namespace]):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        for name, tensor in sorted(saved["model"].items()):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        if digest.hexdigest() != MSA_STATE_SHA256:
            raise ValueError("MSA model tensors differ from the pinned checkpoint")
        if _digest(_regression_path(checkpoint)) != MSA_REGRESSION_SHA256:
            raise ValueError("MSA contact-regression checkpoint differs from the pinned file")
        regression = torch.load(_regression_path(checkpoint), map_location="cpu", weights_only=True)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(MODEL_NAME, saved, regression)
        del saved, regression
        assert model.num_layers == 12 and model.args.embed_dim == 768
        assert alphabet.prepend_bos and not alphabet.append_eos
        # Keep parameter grad flags: changing them alters the screened FP32 forward path.
        model = model.eval().to(device=device, dtype=torch.float32)
        converter = alphabet.get_batch_converter()
        previous_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            for (key, protein), hits in zip(pending.items(), alignments, strict=True):
                rows = diverse_msa(protein.id, protein.seq, hits)
                pooled = _pooled_msa(model, converter, rows, device)
                msa_hash = hashlib.sha256("\n".join(seq for _, seq in rows).encode()).hexdigest()
                with tempfile.NamedTemporaryFile(
                    dir=directory, suffix=".download", delete=False
                ) as f:
                    np.savez_compressed(f, seq=protein.seq, pooled=pooled, msa_hash=msa_hash)
                    temporary = Path(f.name)
                temporary.replace(directory / f"{key}.npz")
        finally:
            torch.set_float32_matmul_precision(previous_precision)
    vectors = []
    for protein in proteins:
        key = hashlib.sha256(protein.seq.encode()).hexdigest()
        with np.load(directory / f"{key}.npz", allow_pickle=False) as saved:
            pooled = saved["pooled"]
            if (
                str(saved["seq"]) != protein.seq
                or pooled.shape != (1536,)
                or pooled.dtype != np.float32
                or not np.isfinite(pooled).all()
            ):
                raise ValueError(f"Invalid MSA cache for {protein.id}")
            vectors.append(pooled.copy())
    return np.stack(vectors)


def self_check() -> None:
    hit = ["q", "target", "0.001", "0.5", "1", "2", "2", "1", "1", "AX", "X-", "XA"]
    assert diverse_msa("q", "AX", [hit, hit]) == [("q", "AX"), ("target", "X-")]
    assert diverse_msa("q", "AX", [[*hit[:11], "AX"]]) == [("q", "AX")]
    assert _window_starts(350) == [0]
    assert _window_starts(1127) == [0, 104]
    counts = np.zeros(3000)
    for start in _window_starts(len(counts)):
        counts[start : start + 1023] += 1
    assert np.all(counts > 0)
