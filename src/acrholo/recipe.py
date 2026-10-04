from .e1 import e1_metadata

ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
ESM_LAYERS = (16, 24, 33)
MSA_MODEL = "esm_msa1b_t12_100M_UR50S"
MSA_STATE_SHA256 = "5931851e2c2d7cc39b8025d619a89d0cffd8dbd2a3c6f859817ebfb532a763c3"
MSA_REGRESSION_SHA256 = "003b2a704d26cf2326a0915bcbc9c8b5aeaaa21fae061052c49302af8156da66"
MMSEQS_VERSION = "057db43b06817bc1f610cba767f4a8dc8e83622f"
UNIREF_RELEASE = "2026_03"
UNIREF_BYTES = 8780552383
UNIREF_MD5 = "0492e3cf4093276ae4319ba24d52514f"
REFERENCE_EXCLUSIONS_SHA256 = "862240f3ee0ae77edc8006fa145bbfdc2f911922a5ab86b855acb36239a7d913"
REFERENCE_SHA256 = {
    "": "69007df4ebe181baeea3f84f28b0438caf869f6884d3e1af0c75e9b94050a1be",
    ".index": "b9abe36a61e07019e4b5e03d8b191f97e10e5b2b61238fd6d5cba4f5f19c9bc0",
    ".dbtype": "c9c7e8c63836a93f6d8c04598b6503d2863a2df7b86366157178234175704dcc",
    "_h": "21a774c34e1add67debde4891ae24776a8f30c311400cd5e789ebfe14f887527",
    "_h.index": "6d0572239ef7b1d5d3185c8ef6706f4b2d516ba023d8055b80f2f35e8ab21024",
    "_h.dbtype": "42f4aeb81c1ef81f771f3de8abca9dcf66901c575530e7672e4b1146474ae650",
}


def feature_recipe(*, query_e1: bool = False) -> dict:
    if query_e1:
        return {
            "representation": "acrholo-query-e1-v1",
            "dimension": 2560,
            "conditioning": "query-only",
            "e1": e1_metadata(),
        }
    return {
        "representation": "acrholo-frozen-four-branch-v1",
        "dimension": 12887,
        "order": ["esm_mean", "esm_population_sd", "msa", "frequency", "available", "e1"],
        "esm": {
            "model": ESM_MODEL,
            "revision": ESM_REVISION,
            "layers": list(ESM_LAYERS),
            "precision": "FP32-high",
            "pooling": "overlap-average-before-mean-populationSD",
        },
        "msa": {
            "model": MSA_MODEL,
            "state_sha256": MSA_STATE_SHA256,
            "regression_sha256": MSA_REGRESSION_SHA256,
            "selection": "query-first-64-accumulated-Hamming-source-order-ties",
            "pooling": "layer12-query-FP32-highest-overlap-average-mean-populationSD",
        },
        "retrieval": {
            "reference": f"UniRef50-{UNIREF_RELEASE}",
            "reference_gzip_md5": UNIREF_MD5,
            "reference_database_sha256": REFERENCE_SHA256.copy(),
            "reference_exclusion_hashes_sha256": REFERENCE_EXCLUSIONS_SHA256,
            "reference_excluded_exact_benchmark_representatives": 821,
            "mmseqs_version": MMSEQS_VERSION,
            "search": "GPU-threads2-E<=0.001-qcov>=0.5-covmode2-max1000",
            "encoding": "B>D,Z>E,J>L,U/O>X;X-preserved",
            "frequency": "top16-nonself-raw-full-sequence-distinct-uniform-pseudocount1",
        },
        "e1": e1_metadata(),
    }
