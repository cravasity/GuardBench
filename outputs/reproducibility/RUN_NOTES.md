# GuardBench execution notes: 2026-09-09

## Analysis contract

The response matrix scores classification items against the canonical human gold
labels in `data/eval_items.jsonl`. No generation judge is introduced. The input
contains 18,841 items. Unsupported model/task combinations are absent, not wrong.
The supported roster contains 27 prompt-harm models, 31 response-harm models, and
six refusal models. Refusal therefore has much less cross-model information for
later IRT estimation; pooling it blindly with the harm tasks is not justified by
the export alone.
Unparsed outputs have nullable correctness in the IRT export. The JSON archives
retain the historical negative-class fallback only in `metrics_conservative`;
`metrics_parsed_only` excludes these failures.

Schema 2.0 retains native labels, per-call outputs, finish reasons, decision-token
positions and top-20 log probabilities. Top-logprob dictionary keys include token
IDs because distinct vocabulary IDs can decode to identical strings. Encoder
classifiers have no decision token: their decision fields are null/empty, and
their raw outputs retain all classifier logits and probabilities instead.

## Verified smoke gates

- Original roster: 21/21 models; 496 item results; zero item/call parse failures;
  zero missing probabilities. Report: `logs/smoke_verified_21.json`.
- Expanded obtainable roster: 32/32 models; 712 item results; zero item/call parse
  failures; zero missing probabilities. Report: `logs/smoke_verified_32.json`.
- The driver revalidates selected models' limit-2 smoke archives before a full run.
  A smoke pass tests format and integration, not classification quality or all
  possible output formats. Full-run failures remain subject to independent audit.
- Sixteen focused unit tests pass, including duplicate decoded tokens, native third
  classes, incomplete multi-call aggregation, GPU memory release, and null IRT
  correctness for failed parses.
- A post-run numerical audit of all 76,635 results from the six encoder models
  reproduced their activations from saved logits (maximum absolute difference
  1.57e-7). All binary predictions and positive probabilities matched the documented
  native-class aggregation rules.
- The final stricter smoke report is `logs/smoke_verified_32_final.json`: 32/32
  models, 712 items, zero parse failures, and zero missing call probabilities. This
  includes fresh GPU smoke tests for the corrected GPT-OSS and GuardReasoner parsers.

## Repairs and compatibility

- GPU reservations are coordinated across workers, including ShieldGemma-27B's
  two-GPU allocation. All reserved GPUs must release memory before the next launch.
- ShieldGemma uses FLASHINFER. Installed flashinfer-python 0.3.1, the version
  specified by vLLM 0.11.0's optional dependency metadata.
- HarmBench uses a 2,048-token context, truncating content while preserving its
  classification rubric and answer prefix.
- DynaGuard uses the upstream Qwen chat format, a continued `<answer>` assistant
  prefix, and strict initial PASS/FAIL parsing. Searching reasoning for these words
  would create false parses.
- GuardReasoner needs the documented 2,048-token output budget and split-token
  recognition for `unharmful`. Its first smoke attempt had a length failure and
  missing probabilities; the corrected rerun passed.
- Its first full-run audit also found 935 outputs with repeated task-label fields,
  with 15 first/final label disagreements. Adapter 1.2 selects the last task-specific
  field for both parsing and decision-token probabilities, including when the final
  field is malformed (no fallback to earlier reasoning). The initial full archives
  are preserved in `outputs/_full_initial_guardreasoner`; a fresh smoke gate passed
  before the full rerun. The validator now checks adapter versions and reparses raw
  output so stale or semantically inconsistent archives cannot be exported.
- SGuard's cached architecture/card identifies five causal decision-token pairs,
  not a conventional sequence-classification head. The HF runner evaluates all five
  pairs using their native safe/unsafe softmax and retains their individual scores.
- Six HF classifier adapters and five previously missing generative adapters are
  implemented. DuoGuard uses its documented Qwen2.5-0.5B tokenizer because the
  classifier snapshot does not include tokenizer files. Only tokenizer/config
  artifacts were downloaded for that base model.
- ShieldLM uses its upstream English evaluation prompt and slow tokenizer.
  SentencePiece 0.2.2 rejected its vocabulary; 0.2.0 loads it correctly.
- GPT-OSS's initial full run revealed 218 final-channel labels that omitted the
  Harmony message separator (one also used a constraint marker). Adapter 1.1
  recognizes these explicit final-channel forms; it does not search reasoning for
  labels. A rerun is required to recover their decision-token probability tables.

The main `.venv` still uses vLLM 0.11.0, torch 2.8.0+cu128, and transformers
4.57.1. Shieldstral's Ministral-3 architecture is unsupported by that stack. Only
Shieldstral uses `.venv-shieldstral` with vLLM 0.26.0, torch 2.11.0, and transformers
5.13.0. Its runtime caches are isolated. The working configuration uses the HF
tokenizer and disables image inputs; the Mistral tokenizer path failed on Pixtral
dummy-image token counts. Both environment freezes are in `logs/environment_*.txt`.
All runtime and dependency writes were directed to `/raid`.

## Interpretation and remaining provenance constraints

Qwen3Guard and ShieldLM native `Controversial` labels are preserved. Their default
binary collapse is strict (positive); alternative collapses must be derived from
the native archives, not guessed from the exported binary correctness column.
GPT-OSS-Safeguard uses an authored general safety policy retained verbatim in each
archive's prompt template; this policy is an evaluation condition, not a gold-label
source. Multi-category maximum probabilities are not calibrated union probabilities.

Nemotron-3.5's actual output omits refusal labels despite its template's instructions.
All four default smoke refusal outputs omitted the label; an explicit output-format
trial still omitted it in three of four. Its refusal task is excluded, rather than
inferred from safety. Failed trial outputs are preserved separately under
`outputs/_unsupported_nemotron_refusal`. Other supported tasks passed smoke.

WildGuard model/dataset access and LlamaGuard-7B approval remain human account actions.
The canonical dataset was not rebuilt or silently switched. Development mirror
provenance, including WildGuardTest, must be resolved before paper results claim
official-source reproduction. No blocked model is represented as evaluated.

## Commands and artifacts

```bash
.venv/bin/python -m unittest discover -s scripts -p test_guardbench.py -v
.venv/bin/python scripts/drive.py --limit 2 --outdir outputs/_smoke --tag smoke
.venv/bin/python scripts/drive.py --gpus 0,1,2,3 --tag full
.venv/bin/python scripts/consolidate_irt.py
```

The full run was launched after the expanded smoke gate passed. Its expected
coverage is 494,270 model-item records across 32 obtainable models. Completion and
observed failure counts must be read from verified full-run artifacts, not inferred
from this expectation. `logs/full_summary.json` is written when the driver exits.
The exporter refuses incomplete or structurally invalid archives, writes
`outputs/irt_long.parquet`, and records input/archive hashes and coverage in
`outputs/irt_long.audit.json`. It permits documented parse failures and preserves
their correctness as null.

## Observed full-run behavior

The initial 32-model sweep completed in 38.5 minutes on GPUs 0-3. It produced all
494,270 expected rows with no runtime or GPU-launch failures. Strict validation
passed 26 models and rejected six for 237 item parse failures. GPT-OSS and
GuardReasoner then received the parser corrections described above and full reruns
after a fresh 2/2-model smoke gate (52 items, no failures).

GPT-OSS's corrected full rerun completed in 314 seconds with zero parse failures
and no missing probabilities. However, it is not a parser-only change to identical
generations: 12,287 of 16,672 raw outputs differed between runs, and 689 of the
16,454 previously parsed items changed binary label. Prompt hashes, input token
counts, model revision, prompt template, dtype, recorded environment, and sampling
settings were identical. Greedy decoding did not ensure repeatability here. The
cause has not been established; numerical/batching effects are a hypothesis, not a
confirmed diagnosis. Both attempts are retained. Repeatability is a remaining
research limitation, distinct from the eliminated generation-judge confound.

SGuard completed all 16,672 items in 956 seconds. Its 83,360 category decisions
were checked against the saved safe/unsafe log probabilities: maximum absolute
probability difference was 8.60e-8. Every generated decision token belonged to its
expected category pair, and all item-level maxima and threshold decisions matched.

## Final verified export

`outputs/irt_long.parquet` contains exactly the requested seven columns and 494,270
unique `(model_key, item_id)` rows from 356 JSON cells, covering 32 models and all
18,841 canonical items. It is 3,219,753 bytes. There are 494,251 parsed responses;
19 rows (0.003844%) have `parse_ok=false`, null correctness, and null probability.
Every parsed row and parsed call has a probability. All 32 archives pass structural,
coverage, metric-recomputation, adapter-version, and raw-output semantic validation.
This does not mean all 32 models have zero parse errors: 27 do, while five do not.

Remaining parse failures:

| Model | Items | Cause |
| --- | ---: | --- |
| GuardReasoner-8B | 8 | Seven 2,048-token reasoning loops; one invalid refusal label |
| Llama Guard 4-12B | 5 | Empty stopped outputs |
| Nemotron Guard v3 | 2 | Missing response-safety field |
| MD-Judge v0.2-7B | 2 | Token cap reached without a result field |
| PolyGuard-Qwen-7B | 2 | Missing required request-label field |

GuardReasoner's final rerun took 813 seconds including loading and validation. Its
raw outputs were identical to the initial run on every item. The parser correction
changed 15 labels and 939 decision-token positions; eight genuine failures remained.
The two-model corrective sweep took 13.6 minutes and overlapped the end of the
initial sweep. Comparisons are saved in `logs/retry_comparison.json`; original and
retry driver summaries are retained separately.

The parquet was independently read back and checked for exact column order, unique
model/item pairs, valid binary correctness, finite probabilities in [0,1], complete
coverage, and exact equivalence between null correctness and failed parsing. The
audit includes per-cell metrics, native-label counts, failure item IDs, finish
reasons, input-limit counts, and SHA-256 hashes of inputs, archives, and source code.
Input-limit counts indicate possible truncation, not a proven exact count of
truncated items. `outputs/reproducibility/` snapshots the implementation, templates,
environment freezes, and these notes. No IRT fit or subset selection has been run.

All eight GPUs were verified idle after inference. WildGuard/LlamaGuard-7B account
access and official dataset provenance remain pending human actions; the canonical
data were not rebuilt during this execution.
