# Model performance scores

Source: ../irt_long.parquet, joined to ../../data/eval_items.jsonl.
One detailed row per model x benchmark x task; both core benchmarks and extras are retained.

- scores_by_model_benchmark.csv: primary scores on parsed responses only.
- f1_by_model_benchmark.csv: the same F1 scores in a model-by-benchmark/task matrix.
- scores_conservative.csv: all items, with unparsed predictions assigned class 0,
  matching the existing archive convention. This is not a uniform failure penalty:
  a failed parse on a negative gold label can count as correct.
- audit.json: source hashes, dimensions, validation checks, and output hashes.

All scores and parse_coverage are fractions in [0,1], not percentages.
Precision, recall, and F1 target harmful content for harm tasks and refusal for
response_refusal. Balanced accuracy is (positive recall + negative recall)/2.
Balanced accuracy and AUROC are blank when both gold classes are not represented.
Precision and F1 use zero for a zero denominator on a nonempty cell; recall and
specificity are blank if their corresponding gold class is absent.
AUROC uses only rows with a recorded probability and handles ties using average ranks.
n_scored_probabilities records its denominator.
Unsupported model/task combinations are blank in the F1 matrix, not zero.
No scores are pooled across benchmarks or tasks, and no IRT ability is estimated.

The parquet omits explicit predictions and gold labels. For parsed binary items,
prediction is gold when correct=1 and 1-gold when correct=0. Predictions are NOT
obtained by thresholding p_positive, because native multi-class aggregation need
not agree with a 0.5 probability threshold. The canonical gold-label source hash
is verified against the parquet metadata before joining.
Scores retain the binary label-collapse policy already used in the parquet.
