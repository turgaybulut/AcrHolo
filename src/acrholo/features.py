from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, EsmModel

from .data import Protein
from .recipe import ESM_LAYERS as LAYERS
from .recipe import ESM_MODEL as MODEL_NAME
from .recipe import ESM_REVISION as MODEL_REVISION

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sequence_key(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:20]


@torch.no_grad()
def prepare_features(
    proteins: list[Protein],
    cache_dir: Path,
    device: str = "cuda",
    batch_size: int = 8,
    *,
    cache_only: bool = False,
) -> list[Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not proteins:
        return []
    config = AutoConfig.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, local_files_only=cache_only
    )
    if (
        config.model_type != "esm"
        or config.num_hidden_layers != 33
        or config.hidden_size != 1280
        or config._commit_hash != MODEL_REVISION
    ):
        raise ValueError("ESM-2 configuration differs from the pinned 650M encoder")
    model_name, revision, layers = MODEL_NAME, MODEL_REVISION, LAYERS
    embedding_dim = config.hidden_size * len(layers)
    cache_version = "overlap-average-v1:layers=16,24,33"
    cache = cache_dir / sequence_key(f"{model_name}:{revision}:{cache_version}")
    cache.mkdir(parents=True, exist_ok=True)
    window_size = min(1022, config.max_position_embeddings - 2)
    residue_start = 1
    special_tokens = 2
    stride = max(1, window_size // 2)
    embeddings = {}
    missing = {}
    windows = []
    for protein in proteins:
        seq = protein.seq
        if not seq:
            raise ValueError(f"{protein.id}: empty sequence")
        if seq in embeddings:
            continue
        path = cache / f"{sequence_key(seq)}.pt"
        if path.exists():
            embedding = torch.load(path, map_location="cpu", weights_only=True)
            if (
                embedding.shape != (len(seq), embedding_dim)
                or embedding.dtype != torch.float32
                or not torch.isfinite(embedding).all()
            ):
                raise ValueError(f"Invalid embedding cache: {path}")
        else:
            if cache_only:
                raise FileNotFoundError(f"Missing required embedding cache: {path}")
            embedding = torch.zeros((len(seq), embedding_dim), dtype=torch.float32)
            missing[seq] = torch.zeros(len(seq), dtype=torch.float32)
            starts = list(range(0, max(1, len(seq) - window_size + 1), stride))
            if starts[-1] + window_size < len(seq):
                starts.append(len(seq) - window_size)
            windows.extend((seq, start, seq[start : start + window_size]) for start in starts)
        embeddings[seq] = embedding
    windows.sort(key=lambda item: len(item[2]))
    if windows:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = (
            EsmModel.from_pretrained(
                model_name, revision=revision, torch_dtype=torch.float32, add_pooling_layer=False
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        offset = 0
        batches = 0
        while offset < len(windows):
            stop = offset + 1
            while stop < len(windows) and stop - offset < batch_size:
                padded_length = len(windows[stop][2]) + special_tokens
                size = stop - offset + 1
                if size * padded_length > 4096 or size * padded_length**2 > 2_000_000:
                    break
                stop += 1
            batch = windows[offset:stop]
            sequences = [window for _, _, window in batch]
            inputs = tokenizer(
                [" ".join(sequence) for sequence in sequences],
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(device)
            if inputs.attention_mask.sum(1).tolist() != [
                len(window) + special_tokens for _, _, window in batch
            ]:
                raise ValueError("Encoder did not produce one token per residue")
            output = model(**inputs, output_hidden_states=True)
            hidden = torch.cat(
                [
                    output.last_hidden_state
                    if layer == config.num_hidden_layers
                    else output.hidden_states[layer]
                    for layer in layers
                ],
                dim=-1,
            )
            hidden = hidden.float().cpu()
            del output
            for row, (seq, start, window) in enumerate(batch):
                embeddings[seq][start : start + len(window)] += hidden[
                    row, residue_start : residue_start + len(window)
                ]
                missing[seq][start : start + len(window)] += 1
            offset = stop
            batches += 1
            if batches % 100 == 0 or offset == len(windows):
                print(
                    f"Encoded {offset}/{len(windows)} residue windows", file=sys.stderr, flush=True
                )
        for seq, counts in missing.items():
            assert torch.all(counts > 0)
            embeddings[seq] /= counts[:, None]
            if not torch.isfinite(embeddings[seq]).all():
                raise ValueError("Encoder returned nonfinite embeddings")
            path = cache / f"{sequence_key(seq)}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(embeddings[seq], temporary)
            temporary.replace(path)
    return [embeddings[p.seq] for p in proteins]


def self_check() -> None:
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    config = SimpleNamespace(
        model_type="esm",
        num_hidden_layers=33,
        hidden_size=1280,
        max_position_embeddings=1026,
        _commit_hash=MODEL_REVISION,
    )
    with TemporaryDirectory() as temporary:
        cache = Path(temporary)
        directory = cache / sequence_key(
            f"{MODEL_NAME}:{MODEL_REVISION}:overlap-average-v1:layers=16,24,33"
        )
        directory.mkdir()
        protein = Protein("synthetic", "AX")
        expected = torch.arange(7680, dtype=torch.float32).reshape(2, 3840)
        path = directory / f"{sequence_key(protein.seq)}.pt"
        torch.save(expected, path)
        with (
            patch.object(AutoConfig, "from_pretrained", return_value=config) as configuration,
            patch.object(AutoTokenizer, "from_pretrained", side_effect=AssertionError),
            patch.object(EsmModel, "from_pretrained", side_effect=AssertionError),
        ):
            rows = prepare_features([protein, protein], cache, cache_only=True)
            assert len(rows) == 2 and torch.equal(rows[0], expected)
            assert rows[0] is rows[1]
            assert configuration.call_args.kwargs["local_files_only"] is True
            try:
                prepare_features([Protein("missing", "M")], cache, cache_only=True)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Missing embedding cache accepted")
            torch.save(expected[:, :1280], path)
            try:
                prepare_features([protein], cache, cache_only=True)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid embedding cache accepted")
