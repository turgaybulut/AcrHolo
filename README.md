# AcrHolo

AcrHolo is a protein-sequence classifier for identifying candidate anti-CRISPR proteins. It combines information about a query sequence with information from related proteins retrieved from UniRef50. Its scores, calibrated probabilities and binary predictions support candidate prioritization for subsequent experimental validation.

The central idea is to represent each query in the context of its homologs. Related proteins contribute aligned residue patterns, amino acid frequency summaries and complete protein chains that E1 processes together with the query. AcrHolo combines these sources of sequence and evolutionary context through four frozen feature branches:

- **ESM-2:** Combines query representations from three layers of the protein language model.
- **MSA Transformer:** Encodes a diverse alignment of the query and its homologs, then pools the query representation.
- **Homolog-frequency features:** Summarize amino acid frequencies and relationships across query positions using retrieved homolog alignments.
- **Homolog-conditioned E1:** Represents the complete query together with whole homolog sequences packed into the model's context.

The four branches form a single feature vector used by a class-balanced logistic classifier. The protein encoders remain frozen during classifier training. A separate reserved training partition supplies probability calibration and a decision threshold targeting 95% positive recall; evaluation labels do not enter fitting, calibration or threshold selection.

The implementation supports reference preparation, feature extraction, training, family-grouped cross-validation, branch ablations and prediction. Encoder outputs are cached, and feature archives can be reused for CPU training and scoring. Train AcrHolo on your labeled dataset, then use the resulting checkpoint to classify new sequences supplied in FASTA or CSV format.

## Setup

Install [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12. Feature extraction requires Linux and an NVIDIA GPU; prediction runs on CPU.

```bash
uv sync
```

## Required data and models

- **Training data:** CSV with `seq`, binary `label` (0 or 1), and optional unique `id` columns. The manuscript uses the [AcrPLMEvo training dataset](https://github.com/Ayanami-0330/AcrPLMEvo/blob/f194993a0977c1752316e5297216308c2da4d8c2/data/anticrispr_benchmarks/anticrispr_binary.train.csv). Save it as `training.csv`.
- **Queries:** FASTA or CSV with `seq` and optional unique `id` columns. Sequences must contain canonical amino acids or `X` and have 1–4,092 residues.
- **Homolog reference:** [UniRef50 2026_03](https://www.uniprot.org/help/downloads) compressed FASTA (MD5 `0492e3cf4093276ae4319ba24d52514f`), saved as `resources/uniref50.fasta.gz`. Use GPU [MMseqs2](https://github.com/soedinglab/MMseqs2) version `057db43b06817bc1f610cba767f4a8dc8e83622f` (CUDA 13 build).
- **MSA Transformer:** Download [MSA-1b](https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt) and its [contact-regression weights](https://dl.fbaipublicfiles.com/fair-esm/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt) into `resources/`, retaining both filenames.
- **E1:** Download the [E1 source](https://github.com/Profluent-AI/E1) at commit `bfd2620a602248499f3d2583d85a7ecddf0b6e02` into `resources/E1` and the [E1-600m model files](https://huggingface.co/Profluent-Bio/E1-600m/tree/52d959fb87a609d15cf223a485127b29ed5c382a) into `resources/E1-600m`. ESM-2 downloads automatically.

Create the separate E1 environment. Do not install optional `flash-attn`.

```bash
uv venv --python 3.12 .e1-venv
uv pip install --python .e1-venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .e1-venv/bin/python numpy transformers==4.56.2 tokenizers==0.22.2 kernels==0.12.3 triton==3.4.0 einops==0.8.2
uv pip install --python .e1-venv/bin/python --no-deps resources/E1
```

## Run

Replace `/path/to/mmseqs` with the GPU MMseqs2 executable. Build the reference once; the command applies the required sequence exclusions and checks the reference contents.

```bash
uv run acrholo reference --input resources/uniref50.fasta.gz --database resources/uniref50_gpu/target --mmseqs /path/to/mmseqs
```

Extract training features and train AcrHolo. Use only training data for this step.

```bash
uv run acrholo features --input training.csv --output outputs/training.npz --database resources/uniref50_gpu/target --mmseqs /path/to/mmseqs --msa-checkpoint resources/esm_msa1b_t12_100M_UR50S.pt --e1-python .e1-venv/bin/python --e1-source resources/E1 --e1-weights resources/E1-600m
uv run acrholo train --input training.csv --features outputs/training.npz --output outputs/acrholo.pt --mmseqs /path/to/mmseqs
```

Extract query features and predict with the trained checkpoint.

```bash
uv run acrholo features --input queries.fasta --output outputs/queries.npz --database resources/uniref50_gpu/target --mmseqs /path/to/mmseqs --msa-checkpoint resources/esm_msa1b_t12_100M_UR50S.pt --e1-python .e1-venv/bin/python --e1-source resources/E1 --e1-weights resources/E1-600m
uv run acrholo predict --features outputs/queries.npz --checkpoint outputs/acrholo.pt --output outputs/predictions.csv
```

Predictions contain `id,score,probability,prediction`. Choose fresh output paths; existing files are not overwritten. For family-grouped cross-validation, run:

```bash
uv run acrholo cross-validate --input training.csv --features outputs/training.npz --output outputs/cv_predictions.csv --mmseqs /path/to/mmseqs
```

Add `--families training_families.csv` to training, cross-validation or ablation to use existing `id,family` assignments.

## Branch ablations

Run the full model, removal of each branch and each branch alone on identical family-held-out folds. Every configuration refits its scaler, classifier and calibrator, with the reserved partition excluded throughout. The command uses the full training feature archive and runs on CPU.

```bash
uv run acrholo ablate --input training.csv --features outputs/training.npz --output outputs/ablations --families training_families.csv
```

To include the query-only E1 control, extract its separate archive using the same training IDs, sequences and order, then supply it with `--query-features`:

```bash
uv run acrholo query-e1 --input training.csv --output outputs/query-e1.npz --e1-python .e1-venv/bin/python --e1-source resources/E1 --e1-weights resources/E1-600m
uv run acrholo ablate --input training.csv --features outputs/training.npz --query-features outputs/query-e1.npz --output outputs/ablations-with-control --families training_families.csv
```

Each fresh output directory contains fold metrics, held predictions, calibration rows, equal-weight fold summaries and absolute/paired intervals. Intervals use 10,000 shared family-bootstrap draws conditional on the fitted models and calibration; they exclude model-selection and refitting uncertainty. Bootstrap requires label-homogeneous families. Failed recall policies remain explicit, with undefined operating-point metrics. These development analyses do not select a replacement predictor or provide independent validation.

For available options and a numerical check:

```bash
uv run acrholo --help
uv run acrholo check
```
