import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
import sys
import tempfile
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import numpy as np

from acrholo.data import read_proteins

MODEL_NAME = "Profluent-Bio/E1-600m"
MODEL_REVISION = "52d959fb87a609d15cf223a485127b29ed5c382a"
SOURCE_REVISION = "bfd2620a602248499f3d2583d85a7ecddf0b6e02"
WEIGHT_SHA256 = "cfc108d4b98baaa62932331b40be265eae39dc382595bc3cde4a5ab55db1bf7a"
CODE_SHA256 = "ca314d2b203b6203df804424551493c5c57008533a938de0c3cc5347fd052b8f"
CONFIG_SHA256 = "0aba9a66d5293ccc014dc07695a2e880a0124c26b5407e25d0d46250a3aa39c9"
TOKENIZER_SHA256 = "4430b4c38556d66807f00c817aa20a100bbc1058e1bbe84eec679d5265ceccd1"
RUNTIME = {
    "torch": "2.8.0+cu128",
    "transformers": "4.56.2",
    "tokenizers": "0.22.2",
    "kernels": "0.12.3",
    "triton": "3.4.0",
    "einops": "0.8.2",
}
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
NORMALIZATION = "query-canonical20+X;target-B>D,Z>E,J>L,U/O>X;X-preserved"
RECIPE = (
    "E<=.001;qcov>=.5;normalized-query-remove;full-normalized-sequence-dedup;"
    "sort-evalue-targetID;greedy-whole-chains-skip-oversize;4096-total-L+4;"
    "max511-homologs;query-last;BF16-parameters-FP32-rotary;native-Flex-FAST_BLOCK_MASK1;"
    "offline-native-RMSNorm;no-KV-cache;eval-no_grad;batch1-highest;"
    "final-RMS-query-residues;CPU-FP32-mean-populationSD2560;no-L2"
)


def _sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def e1_metadata() -> dict:
    return {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "code_sha256": CODE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "runtime": {**RUNTIME, "cpu_threads": 2},
        "normalization": NORMALIZATION,
        "recipe": RECIPE,
    }


def e1_cache_directory(cache_dir: Path, metadata: dict, alignment_dir: Path | None = None) -> Path:
    identity = {
        **metadata,
        "cache_format": "acrholo-e1-v1",
        "retrieval_namespace": alignment_dir.name if alignment_dir else "",
    }
    directory = cache_dir / _sha(json.dumps(identity, sort_keys=True))[:20]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def pack_e1(query: str, hits: list[list[str]]) -> list[str]:
    if not query or set(query) - (CANONICAL | {"X"}) or len(query) + 4 > 4096:
        raise ValueError("E1 requires a complete canonical20+X query fitting 4096 total tokens")
    candidates = []
    for row in hits:
        if len(row) != 12 or row[0] != _sha(query) or not row[1] or not row[11]:
            raise ValueError("Invalid query-local full-target alignment row")
        evalue, identity, qcov, tcov = map(float, (row[2], row[3], row[7], row[8]))
        begin, end, length = map(int, row[4:7])
        if (
            not all(math.isfinite(value) for value in (evalue, identity, qcov, tcov))
            or evalue < 0
            or not all(0 <= value <= 1 for value in (identity, qcov, tcov))
            or length != len(query)
            or not 1 <= begin <= end <= length
            or len(row[9]) != len(row[10])
            or row[9].replace("-", "") != query[begin - 1 : end]
            or set(row[11]) - (CANONICAL | set("XBZJUO"))
        ):
            raise ValueError("Invalid full query/target sequence, coordinates or significance")
        target = row[11].translate(
            str.maketrans({"B": "D", "Z": "E", "J": "L", "U": "X", "O": "X"})
        )
        if evalue <= 0.001 and qcov >= 0.5:
            candidates.append((evalue, row[1], target))
    chains = []
    seen = {query}
    tokens = len(query) + 4
    for _, _, target in sorted(candidates, key=lambda row: (row[0], row[1])):
        if target in seen:
            continue
        seen.add(target)
        if tokens + len(target) + 4 > 4096:
            continue
        chains.append(target)
        tokens += len(target) + 4
        if len(chains) == 511:
            break
    return [*chains, query]


def _inputs(sequences: list[str], alignment_dir: Path | None) -> dict[str, tuple[list[str], str]]:
    result = {}
    for query in sequences:
        if query in result:
            continue
        pack_e1(query, [])
        hits = []
        fingerprint = ""
        if alignment_dir is not None:
            path = alignment_dir / f"{_sha(query)}.hits.npz"
            content = path.read_bytes()
            fingerprint = hashlib.sha256(content).hexdigest()
            with np.load(io.BytesIO(content), allow_pickle=False) as saved:
                rows = saved["hits"]
                if (
                    saved["seq"].shape != ()
                    or str(saved["seq"]) != query
                    or rows.ndim != 2
                    or rows.shape[1] != 12
                    or rows.dtype.kind != "U"
                ):
                    raise ValueError(f"Invalid alignment cache: {path}")
                hits = rows.tolist()
        result[query] = (pack_e1(query, hits), fingerprint)
    return result


def _read(path: Path, query: str, chains: list[str], hits_sha256: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        pooled = saved["pooled"]
        if (
            saved["cache_identity"].shape != ()
            or str(saved["cache_identity"]) != path.parent.name
            or str(saved["query"]) != query
            or str(saved["query_sha256"]) != _sha(query)
            or str(saved["pack_sha256"]) != _sha(",".join(chains))
            or str(saved["hits_sha256"]) != hits_sha256
            or saved["chains"].dtype.kind != "U"
            or saved["chains"].tolist() != chains
            or pooled.shape != (2560,)
            or pooled.dtype != np.float32
            or not np.isfinite(pooled).all()
        ):
            raise ValueError(f"Invalid or stale E1 representation cache: {path}")
        return pooled.copy()


def _save(
    directory: Path, query: str, chains: list[str], hits_sha256: str, pooled: np.ndarray
) -> None:
    if pooled.shape != (2560,) or pooled.dtype != np.float32 or not np.isfinite(pooled).all():
        raise ValueError("E1 pooling requires 2560 finite float32 values")
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as handle:
        np.savez_compressed(
            handle,
            cache_identity=directory.name,
            query=query,
            query_sha256=_sha(query),
            pack_sha256=_sha(",".join(chains)),
            hits_sha256=hits_sha256,
            chains=np.asarray(chains),
            pooled=pooled,
        )
        temporary = Path(handle.name)
    temporary.replace(directory / f"{_sha(query)}.npz")


def _cached(
    directory: Path,
    cache_dir: Path,
    metadata: dict,
    query: str,
    chains: list[str],
    fingerprint: str,
) -> np.ndarray | None:
    path = directory / f"{_sha(query)}.npz"
    if path.exists():
        return _read(path, query, chains, fingerprint)
    if len(chains) == 1 and fingerprint:
        query_directory = e1_cache_directory(cache_dir, metadata)
        query_path = query_directory / path.name
        if query_path.exists():
            pooled = _read(query_path, query, [query], "")
            _save(directory, query, chains, fingerprint, pooled)
            return pooled
    return None


def prepare_e1(
    sequences: list[str], cache_dir: Path, metadata: dict, alignment_dir: Path | None = None
) -> np.ndarray:
    directory = e1_cache_directory(cache_dir, metadata, alignment_dir)
    vectors = {}
    for query, (chains, fingerprint) in _inputs(sequences, alignment_dir).items():
        pooled = _cached(directory, cache_dir, metadata, query, chains, fingerprint)
        if pooled is None:
            raise FileNotFoundError(
                f"Missing E1 cache for {_sha(query)}; "
                "run python -m acrholo.e1 with the isolated E1 interpreter"
            )
        vectors[query] = pooled
    return (
        np.stack([vectors[query] for query in sequences])
        if sequences
        else np.empty((0, 2560), dtype=np.float32)
    )


def _verify_native(source: Path, weights: Path) -> dict:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("E1 extraction requires Python 3.12")
    if {name: version(name) for name in RUNTIME} != RUNTIME:
        raise ValueError("E1 extraction runtime differs from the accepted native recipe")
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("FAST_BLOCK_MASK", "1") != "1"
        or find_spec("flash_attn") is not None
    ):
        raise ValueError("Use offline native RMSNorm/FlexAttention without optional flash-attn")
    specification = find_spec("E1")
    if specification is None or specification.submodule_search_locations is None:
        raise ValueError("Install the pinned native E1 package in the isolated interpreter")
    installed = Path(next(iter(specification.submodule_search_locations)))
    native = source / "src/E1"
    paths = sorted(native.rglob("*.py"))
    if {path.relative_to(native) for path in paths} != {
        path.relative_to(installed) for path in installed.rglob("*.py")
    }:
        raise ValueError("Installed E1 Python files differ from the supplied native source")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(native)
        content = path.read_bytes()
        if content != (installed / relative).read_bytes():
            raise ValueError("Installed E1 code differs from the pinned native source")
        digest.update(relative.as_posix().encode())
        digest.update(hashlib.sha256(content).digest())
    if (
        digest.hexdigest() != CODE_SHA256
        or _digest(installed / "tokenizer.json") != TOKENIZER_SHA256
    ):
        raise ValueError("E1 source/tokenizer identity differs from the accepted pin")
    if (
        _digest(weights / "config.json") != CONFIG_SHA256
        or _digest(weights / "model.safetensors") != WEIGHT_SHA256
    ):
        raise ValueError("E1 checkpoint differs from the accepted immutable revision")
    import torch

    if (
        not torch.cuda.is_available()
        or torch.version.cuda != "12.8"
        or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        raise ValueError("E1 requires CUDA 12.8 and a visible CUDA GPU with native BF16 support")
    torch.set_num_threads(2)
    return {
        **e1_metadata(),
        "runtime": {
            **RUNTIME,
            "python": ".".join(map(str, sys.version_info[:3])),
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
        },
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": list(torch.cuda.get_device_capability()),
    }


def extract_e1(
    sequences: list[str],
    cache_dir: Path,
    metadata: dict,
    weights: Path,
    alignment_dir: Path | None = None,
) -> np.ndarray:
    directory = e1_cache_directory(cache_dir, metadata, alignment_dir)
    pending = []
    for query, (chains, fingerprint) in _inputs(sequences, alignment_dir).items():
        if _cached(directory, cache_dir, metadata, query, chains, fingerprint) is None:
            pending.append((query, chains, fingerprint))
    if not pending:
        return prepare_e1(sequences, cache_dir, metadata, alignment_dir)
    import torch

    modeling = importlib.import_module("E1.modeling")
    preparer_class = importlib.import_module("E1.batch_preparer").E1BatchPreparer
    if modeling.layer_norm is not None:
        raise ValueError("E1 requires offline native RMSNorm")
    torch.set_num_threads(2)
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    network = None
    try:
        network = (
            modeling.E1ForMaskedLM.from_pretrained(
                weights, torch_dtype=torch.bfloat16, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )
        cast(torch.nn.Module, network).to("cuda")
        assert sum(p.numel() for p in network.parameters()) == 641438754
        assert all(
            p.dtype == torch.bfloat16 and not p.requires_grad and p.device.type == "cuda"
            for p in network.parameters()
        )
        assert len(list(network.buffers())) == 90
        assert all(b.dtype == torch.float32 for b in network.buffers())
        preparer = preparer_class(device=torch.device("cuda"))
        for completed, (query, chains, fingerprint) in enumerate(pending, 1):
            query_directory = e1_cache_directory(cache_dir, metadata)
            query_path = query_directory / f"{_sha(query)}.npz"
            if len(chains) == 1 and fingerprint and query_path.exists():
                pooled = _read(query_path, query, [query], "")
            else:
                batch = preparer.get_batch_kwargs([",".join(chains)], device=torch.device("cuda"))
                tokens = {
                    name: batch[name]
                    for name in (
                        "input_ids",
                        "within_seq_position_ids",
                        "global_position_ids",
                        "sequence_ids",
                    )
                }
                assert tokens["input_ids"].shape == (1, sum(len(chain) + 4 for chain in chains))
                assert tokens["sequence_ids"].max().item() == len(chains) - 1
                mask = (
                    tokens["sequence_ids"] == len(chains) - 1
                ) & ~preparer.get_boundary_token_mask(tokens["input_ids"])
                assert tokens["input_ids"][mask].tolist() == [
                    preparer.vocab[residue] for residue in query
                ]
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    output = network(
                        **tokens,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                assert output.embeddings.dtype == torch.bfloat16
                nodes = output.embeddings[mask].to(device="cpu", dtype=torch.float32)
                assert nodes.shape == (len(query), 1280) and torch.isfinite(nodes).all()
                pooled = torch.cat((nodes.mean(0), nodes.std(0, correction=0))).numpy()
                del output, nodes, batch, tokens, mask
                if len(chains) == 1:
                    _save(query_directory, query, [query], "", pooled)
            _save(directory, query, chains, fingerprint, pooled)
            if completed % 25 == 0 or completed == len(pending):
                print(
                    f"E1 extracted {completed}/{len(pending)} queries; "
                    f"query_length={len(query)}, chains={len(chains)}, "
                    f"tokens={sum(len(chain) + 4 for chain in chains)}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        del network
    return prepare_e1(sequences, cache_dir, metadata, alignment_dir)


def self_check() -> None:
    query = "ACX"
    metadata = {**e1_metadata(), "runtime": {"synthetic_check": True}}

    def hit(identifier: str, target: str, evalue: str = "0.0001", coverage: str = "1") -> list[str]:
        return [
            _sha(query),
            identifier,
            evalue,
            "0.5",
            "1",
            "3",
            "3",
            coverage,
            "1",
            query,
            "ACX",
            target,
        ]

    hits = [
        hit("z", "M" * 4096),
        hit("b", "BZJUO"),
        hit("a", "DELXX"),
        hit("self", query),
        hit("skip", "AAA", coverage="0.4"),
    ]
    assert pack_e1(query, hits) == ["DELXX", query]
    assert pack_e1(query, list(reversed(hits))) == ["DELXX", query]
    assert pack_e1(query, []) == [query]
    assert pack_e1(query, [hit("a", "M" * 4096), hit("b", "M")]) == ["M", query]
    assert pack_e1(query, [hit("inclusive", "M", evalue="0.001", coverage="0.5")]) == ["M", query]
    assert pack_e1(query, [hit("excluded", "M", evalue="0.0011")]) == [query]
    targets = [
        "ACDEFGHIKLMNPQRSTVWY"[i // 400]
        + "ACDEFGHIKLMNPQRSTVWY"[(i // 20) % 20]
        + "ACDEFGHIKLMNPQRSTVWY"[i % 20]
        for i in range(512)
    ]
    assert len(pack_e1(query, [hit(f"{i:03}", target) for i, target in enumerate(targets)])) == 512
    assert pack_e1("X" * 4092, []) == ["X" * 4092]
    for invalid_query in ("", "J", "X" * 4093):
        try:
            pack_e1(invalid_query, [])
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid query accepted")
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)
        alignments = cache / "retrieval"
        alignments.mkdir()
        np.savez(
            alignments / f"{_sha(query)}.hits.npz", seq=query, hits=np.empty((0, 12), dtype=str)
        )
        vector = np.arange(2560, dtype=np.float32)
        _save(e1_cache_directory(cache, metadata), query, [query], "", vector)
        original = prepare_e1([query, query], cache, metadata)
        fallback = prepare_e1([query, query], cache, metadata, alignments)
        assert original.dtype == np.float32 and original.shape == (2, 2560)
        assert np.array_equal(original, fallback)
        try:
            prepare_e1(["M"], cache, metadata)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing cache accepted")
        directory = e1_cache_directory(cache, metadata)
        _save(directory, query, [query], "", vector)
        with np.load(directory / f"{_sha(query)}.npz", allow_pickle=False) as saved:
            invalid = dict(saved)
        invalid["pack_sha256"] = "wrong"
        np.savez(directory / f"{_sha(query)}.npz", **invalid)
        try:
            prepare_e1([query], cache, metadata)
        except ValueError:
            pass
        else:
            raise AssertionError("Stale pack accepted")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pinned native E1 pooled representation extraction"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, help="Omit for query-only E1 extraction")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    with arguments.input.open() as handle:
        if csv.DictReader(handle).fieldnames != ["id", "seq"]:
            raise ValueError("Provide a label-free id,seq query CSV")
    sequences = [protein.seq for protein in read_proteins(arguments.input)]
    metadata = _verify_native(arguments.source, arguments.weights)
    print(
        f"E1 verified runtime: {json.dumps(metadata['runtime'], sort_keys=True)}; "
        f"device={metadata['cuda_device']}; capability={metadata['cuda_capability']}",
        file=sys.stderr,
        flush=True,
    )
    if arguments.cache_only:
        pooled = prepare_e1(sequences, arguments.cache, metadata, arguments.alignments)
    else:
        pooled = extract_e1(
            sequences, arguments.cache, metadata, arguments.weights, arguments.alignments
        )
    with arguments.output.open("xb") as handle:
        np.save(handle, pooled, allow_pickle=False)
    print(
        f"E1 wrote {len(sequences)} pooled float32 vectors to {arguments.output}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
