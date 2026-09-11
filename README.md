# GuardBench

Research code for constructing a condensed, high-discrimination LLM-safety
benchmark from open-weight guard-model evaluations and Item Response Theory
(IRT, planned 2PL fitting).

**The observation is a classification item scored against a human reference
label.** We do not generate answers and have an LLM judge score them. Guard
reasoning, when present, is only part of the classifier's output. This removes
the generation-judge confound, not the possibility of noise in human labels.

## Current status

Snapshot: September 9, 2026.

- 32 obtainable guard models evaluated on 18,841 canonical items.
- 10 benchmarks: eight core benchmarks and two extras, comprising 14
  benchmark-by-task cells.
- 356 model-by-benchmark-by-task JSON archives; 494,270 unique model-item rows.
- 494,251 parsed results; 19 parse failures have null correctness and probability.
- All parsed results and parsed calls have probabilities.
- Expanded smoke test: 32/32 models, 712 item results, zero parse failures.
- IRT fitting, final item curation, and condensed-subset selection are **not done**.

The data is being retained as collected. Known duplicate inputs, conflicting
labels, and repeatability limitations are documented below, not silently fixed.
See [RUN_NOTES.md](RUN_NOTES.md) for verified execution results and repairs.
[HANDOFF.md](HANDOFF.md) includes the original project contract, but its historical
pre-repair sections describe problems that have since been fixed. In particular,
SGuard uses causal decision-token pairs, not a conventional classification head.

## Start here

- **Analyze results:** read `outputs/irt_long.parquet` and join
  `data/eval_items.jsonl` by `item_id`.
- **Inspect a prediction:** open the corresponding
  `outputs/{model_key}__{benchmark}__{task}.json`.
- **Understand model handling:** read `scripts/models_manifest.py`,
  `scripts/adapters.py`, and `scripts/hf_cls.py`.
- **Check provenance and validity:** read `data/provenance.json`,
  `outputs/irt_long.audit.json`, and `RUN_NOTES.md`.

## Where the data lives

The GitHub repository contains code, templates, documentation, source metadata,
environment freezes, canonical benchmark inputs, prediction archives, Parquet
results, score reports, and execution logs. This is a collection snapshot, not
the final curated benchmark. **Model weights, environments, runtime caches, and
credentials are excluded.** Upstream data and model terms still apply.

Per-model/benchmark/task scores are in
[`outputs/model_scores/`](outputs/model_scores/). The primary CSV excludes parse
failures and reports coverage; a separate conservative CSV uses the historical
negative-class fallback. See its README for metric definitions.

Local backup files are retained as historical artifacts, not evaluation inputs.
In particular, `outputs/aprielguard-8b__aegis_v1__prompt_harm.json.bak` contains
trailing data and is not valid JSON; it is preserved without modification. The
current `.json` matches the recorded Parquet source-archive hash. Score exports
were not regenerated as part of publishing this snapshot.

The existing artifacts are on the project DGX-H100, under:

```text
/raid/MLP/hgkim/guardbench/
  data/eval_items.jsonl       Canonical inputs and reference labels (~19 MB)
  data/provenance.json        Source repositories, files, and mirror status
  outputs/irt_long.parquet    Analysis table (~3.2 MB; 494,270 rows)
  outputs/irt_long.audit.json Coverage, failures, metrics, and SHA-256 hashes
  outputs/*.json             Current full-run schema-2.0 prediction archives
  outputs/_smoke/            Small integration-test archives, not full results
  outputs/_full_initial_gptoss/        Preserved earlier GPT-OSS attempt
  outputs/_full_initial_guardreasoner/ Preserved earlier GuardReasoner attempt
  outputs/_unsupported_nemotron_refusal/  Unsupported-task trial outputs
  outputs/reproducibility/   Source/template/environment snapshots and hashes
  logs/                     Driver summaries, validation reports, and run logs
  models/                   Download/probe status; NOT the model weights
  runtime/                  Temporary files and compilation/runtime caches
```

Weights are in the shared Hugging Face cache:
`/raid/MLP/.cache/huggingface` (approximately 422.6 GB downloaded for this project).
Do not delete or reorganize this shared cache.

For analysis, the small Parquet file,
canonical JSONL, provenance, and audit are the starting bundle; the raw JSON
archives are additionally needed for native-label or parsing investigations.
Benchmark content includes harmful language. Check upstream licenses and access
conditions before copying or redistributing it. Re-running inference requires
appropriate compute access and separately authorized model downloads.

## Code map

| File | Responsibility |
| --- | --- |
| `scripts/models_manifest.py` | Model keys, repositories, families, sizes, runner selection, blocked models |
| `scripts/build_datasets.py` | Build canonical classification items and source metadata |
| `scripts/adapters.py` | Generative guard prompts, native-label parsing, decision vocabularies, aggregation |
| `scripts/runner.py` | vLLM inference, decision-token logprobs, metrics, schema-2.0 writer; dispatches HF models |
| `scripts/hf_cls.py` | Six encoder classifiers plus SGuard's causal category-token scoring |
| `scripts/drive.py` | GPU scheduling, memory-release checks, per-model settings, smoke gate |
| `scripts/validate_outputs.py` | Coverage, labels, metrics, parser consistency, and probability checks |
| `scripts/consolidate_irt.py` | Validate full archives and export nullable correctness to Parquet |
| `scripts/test_guardbench.py` | Focused unit tests for parsing, probabilities, scheduling, and export |
| `scripts/download_models.py` | Download available model snapshots into the shared cache |
| `scripts/probe_models.py`, `scripts/probe_datasets.py` | Availability/schema investigation helpers |
| `scripts/extract_templates.py` | Extract model-specific template assets from cached models |
| `scripts/calibrate.py` | Throughput measurement helper |
| `scripts/make_summary.py` | Historical progress-report generator with a separate hardcoded output directory |
| `templates/` | ShieldGemma policies and Nemotron template assets |

This is a research snapshot, not an installable package. Several scripts hardcode
the DGX project path and cache path. A clone elsewhere needs path configuration
work before execution; a different current working directory does not redirect
these paths. Auxiliary scripts may write files when run or imported.

## Data schemas

### Canonical items: `data/eval_items.jsonl`

One JSON object per classification item. Synthetic example, not benchmark content:

```json
{"item_id":"example::prompt_harm::0","benchmark":"example","task":"prompt_harm","prompt":"A synthetic benign request.","response":"","label":0,"label_str":"unharmful","core":true,"meta":{}}
```

| Field | Meaning |
| --- | --- |
| `item_id` | Unique join key within this canonical snapshot |
| `benchmark`, `task` | Dataset and classification task |
| `prompt`, `response` | Input text; response is empty for prompt-only tasks |
| `label`, `label_str` | Binary reference label and its textual representation |
| `core` | Whether this benchmark belongs to the core suite |
| `meta` | Dataset-specific annotations, sometimes empty; not a uniform taxonomy |

For `prompt_harm` and `response_harm`, positive (`1`) means harmful. For
`response_refusal`, positive means refusal. Correctness is a different quantity:
a model can be correct by predicting either class.

IDs are assigned by the builder after task-specific filtering. Do not use their
numeric suffixes as original-source row IDs or join conversations across tasks
by suffix. Unique IDs also do not imply unique input text.

### Analysis table: `outputs/irt_long.parquet`

Exactly seven columns, one row per supported `(model_key, item_id)` pair:

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `model_key` | string | Model identifier from the manifest |
| `item_id` | string | Join key into canonical items |
| `benchmark` | string | Benchmark identifier |
| `task` | string | Classification task |
| `correct` | nullable int8 | 1 if parsed binary prediction equals gold, 0 otherwise; null on parse failure |
| `p_positive` | nullable float64 | Model-specific harmfulness/refusal score, not correctness probability |
| `parse_ok` | bool | Whether an item-level binary prediction was recovered |

Unsupported model-task combinations are absent, not incorrect. Parse failures
are missing responses for IRT, not zeros. Feed `correct` into binary IRT, not
`p_positive`. The latter is heterogeneous across generative decision-token
normalization and classifier/category aggregation; it is not the fitted IRT
probability in the Fisher-information formula.

### Prediction archives: schema 2.0

One file per model, benchmark, and task. Top-level fields record `schema_version`,
`run`, `model`, `benchmark`, `task`, `positive_class`, `sampling`, `item_source`,
`metrics`, `n_items`, and `results`.

- `run`: timestamp, host/GPU, timing, and software environment.
- `model`: repository, resolved revision SHA, adapter/version, prompt template,
  native-to-binary label mapping, and model metadata.
- Each `results` entry: `item_id`, `label_gt`, `label_gt_str`, `pred`, `pred_raw`,
  `pred_native`, `parse_ok`, `p_positive`, `categories`, `aggregation`, `n_calls`,
  and `calls`.
- `pred_raw` is null on parse failure. `pred` uses the historical negative-class
  fallback for conservative metrics; do not use it to score failed parses in IRT.
- `pred_native` preserves third classes such as `Controversial`. The existing
  Qwen3Guard/ShieldLM collapse is positive; alternative policies must be derived
  from the raw archive, not the already-collapsed Parquet correctness.
- `calls` is always a list: ShieldGemma has four policy calls per item. Calls
  retain raw output, native/binary verdicts, scores, `finish_reason`, prompt hash,
  token counts, and `decision`.
- `decision` holds the token index, decoded token, and top-20 logprob table.
  Table keys include token IDs to distinguish identical decoded strings.
  Encoder classifiers have no decision token; their raw output stores classifier
  logits/probabilities instead. SGuard retains five category-token decisions.
- `metrics.metrics_conservative` uses the negative fallback on parse failure;
  `metrics.metrics_parsed_only` excludes failures. Both are retained alongside
  parse counts and error rate.

The JSON files are the detailed evidence; the Parquet file is the compact analysis
view. Item text is deliberately not duplicated into every prediction record.

## Read the results without inference

On the DGX, use the existing `.venv/bin/python`. On an analysis-only machine,
`pandas` and `pyarrow` suffice for this example once the data has been transferred.
No model weights or GPU are needed.

```python
from pathlib import Path
import pandas as pd

root = Path('/raid/MLP/hgkim/guardbench')  # Or your approved local data copy.
scores = pd.read_parquet(root / 'outputs/irt_long.parquet')
items = pd.read_json(root / 'data/eval_items.jsonl', lines=True)
joined = scores.merge(
    items[['item_id', 'label', 'core', 'meta']],
    on='item_id', how='left', validate='many_to_one',
)
cell = joined.loc[
    (joined['benchmark'] == 'wildguardtest')
    & (joined['task'] == 'prompt_harm')
]
matrix = cell.pivot(index='model_key', columns='item_id', values='correct')
observed = matrix.notna()  # Preserve missingness; do not fill with zero.
print(matrix.shape)
```

Do not recursively combine every JSON under `outputs/`: that would mix smoke,
historical attempts, and unsupported-task trials with the current full run.

## Coverage

| Benchmark | Prompt harm | Response harm | Response refusal | Core |
| --- | ---: | ---: | ---: | --- |
| WildGuardTest | 1,699 | 1,709 | 1,720 | Yes |
| ToxicChat | 2,853 | - | - | Yes |
| OpenAI moderation | 1,680 | - | - | Yes |
| Aegis v1 | 359 | - | - | Yes |
| HarmBench | 239 | 602 | - | Yes |
| BeaverTails | - | 3,021 | - | Yes |
| SafeRLHF | - | 2,000 | - | Yes |
| XSTest-Response | - | 446 | 449 | Yes |
| SimpleSafetyTests | 100 | - | - | No |
| Aegis v2 | 1,964 | - | - | No |
| Total items | 8,894 | 7,778 | 2,169 | |
| Supported models | 27 | 31 | 6 | |

There are 32 distinct evaluated models, not 32 models for every task. Only five
cover all three tasks. Model variants can share backbones/training, so these are
not 32 independent model families. Nemotron-3.5 refusal is unsupported in this
evaluation; refusal is not inferred from its safety label.

## Existing environment and commands

The measured environment is an 8 x H100 80 GB DGX, Python 3.12. Use the existing
environments on that machine; do not upgrade them in place.

| Environment | vLLM | torch | transformers |
| --- | --- | --- | --- |
| `.venv` | 0.11.0 | 2.8.0+cu128 | 4.57.1 |
| `.venv-shieldstral` (Shieldstral only) | 0.26.0 | 2.11.0 | 5.13.0 |

The complete installed-package snapshots are
[environment_main.txt](logs/environment_main.txt) and
[environment_shieldstral.txt](logs/environment_shieldstral.txt). These are observed
freezes, not a tested portable installer. Transformers 5.x breaks the main vLLM
0.11 stack. ShieldGemma uses FLASHINFER; ShieldLM needs SentencePiece 0.2.0 in
the main environment. The driver handles these model-specific settings.

```bash
cd /raid/MLP/hgkim/guardbench
export HF_HOME=/raid/MLP/.cache/huggingface
export HF_TOKEN="$HF_MJ_READ_TOKEN"  # DGX's existing authorized HF token variable.
export TMPDIR=/raid/MLP/hgkim/guardbench/runtime/tmp
mkdir -p "$TMPDIR"

# CPU-only unit tests; no inference.
.venv/bin/python -m unittest discover -s scripts -p test_guardbench.py -v

# Read-only strict validation of the existing smoke archives.
.venv/bin/python scripts/validate_outputs.py --outdir outputs/_smoke --limit 2
```

Use your own authorized Hugging Face token if the DGX token variable is not
available. Never put a token literal in code, documentation, or Git.

Inference commands below are optional and **write outputs**. Coordinate GPU use
first. A tag changes log filenames, not the prediction destination: always choose
a new `--outdir` for a new full attempt to preserve existing results.

```bash
# Replaces smoke files: 2 items per benchmark/task, not 2 items per model.
.venv/bin/python scripts/drive.py --gpus 0,1,2,3 --limit 2 --outdir outputs/_smoke --tag smoke_new

# Full run requires the selected models to pass the existing _smoke gate.
.venv/bin/python scripts/drive.py --gpus 0,1,2,3 --outdir outputs/full_new --tag full_new

# Export a completed new run without overwriting the current analysis table.
.venv/bin/python scripts/consolidate_irt.py --outdir outputs/full_new --output outputs/full_new/irt_long.parquet
```

The driver/standalone validator are strict about parse failures: they can report
failure despite completed inference. The exporter permits documented parse
failures with null correctness, but rejects incomplete/invalid archives. Check
the audit rather than treating process completion as proof of valid predictions.

Dataset rebuilding and model downloading are not needed for the current analysis.
`build_datasets.py` overwrites canonical inputs/provenance and can switch sources
when gated access changes. Do not rebuild beneath existing results casually.
It also uses the separate `allenai/safety-eval` checkout, excluded from this repo;
the observed checkout revision was
`060cc903d64703214c549b5c3a30ea8ceef2e588`. Restoring that checkout alone does not
pin all remote datasets: many recorded dataset revisions are null.

All runtime/dependency writes should remain on `/raid`. The helper
`make_summary.py` has a historical output path under `/home`; inspect it before
use rather than assuming the driver controls every helper's destination.

## Known limitations before IRT

- **Input duplication and label conflicts:** exact `(task, prompt, response)`
  grouping found 737 duplicate groups (835 extra rows), including 80 groups with
  conflicting binary gold labels. Thirty conflicts occur within a benchmark:
  28 in SafeRLHF and two in Aegis v2. The source of these disagreements has not
  been resolved. No deduplication or relabeling has been applied.
- **Benchmark dependence:** every Aegis v1 input occurs in Aegis v2. Their ability
  correlation is not evidence from independent item sets. Cross-dataset label
  disagreements may reflect different policies, not extraction errors.
- **Limited calibration roster:** refusal has only six models. Across all tasks,
  5,793 items have unanimous observed correctness or incorrectness. Assess IRT
  identifiability, regularization, and uncertainty before interpreting parameters.
- **True parse failures remain:** GuardReasoner 8, Llama Guard 4 5, Nemotron v3 2,
  MD-Judge 2, PolyGuard 2. Do not conceal these with fallback predictions in IRT.
- **Repeatability:** GPT-OSS changed 689 of 16,454 previously parsed labels on a
  rerun with identical recorded prompts/settings/revision. The cause is unknown.
  Both attempts and `logs/retry_comparison.json` are retained in this snapshot.
- **Native label/policy sensitivity:** third-class collapse, authored GPT-OSS
  policy, model-specific truncation, and category aggregation affect the response
  matrix. Guard confidence scores are not universally calibrated probabilities.
- **Gated access/provenance:** WildGuard and original LlamaGuard-7B were not
  evaluated. WildGuardTest currently uses `iagoalves/wildguardmix_eval`;
  XSTest-Response uses the original English fields in
  `heegyu/xstest-response-ko`. Official-source access and dataset revision pinning
  need resolution before claims of official-source paper reproduction.

Validation establishes archive/schema consistency against the canonical labels,
not that the labels are error-free or the items statistically independent.
Preserve this collection snapshot when creating future curated analysis views.
Models, datasets, and upstream template assets retain their respective terms;
publishing this code does not grant permission to redistribute those resources.
