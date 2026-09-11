"""Build a complete long-format IRT table from schema-2.0 classification archives."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

from models_manifest import MODELS
from runner import D, load_items
from validate_outputs import supported_tasks, validate_model


def consolidate(outdir, destination, keys):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Model keys must be nonempty and unique")
    outdir, destination = Path(outdir), Path(destination)
    source_items = load_items()
    items = {it["item_id"]: it for it in source_items}
    if len(items) != len(source_items):
        raise ValueError("Duplicate canonical item IDs")
    validation = {key: validate_model(key, outdir, strict_parsing=False, require_probabilities=False) for key in keys}
    failures = {key: report for key, report in validation.items() if not report["ok"]}
    if failures:
        raise ValueError("Incomplete or invalid archive: " + json.dumps(failures))
    schema = pa.schema([
        ("model_key", pa.string()), ("item_id", pa.string()), ("benchmark", pa.string()),
        ("task", pa.string()), ("correct", pa.int8()), ("p_positive", pa.float64()), ("parse_ok", pa.bool_()),
    ], metadata={
        b"unit": b"classification item scored against canonical human gold label",
        b"parse_failure": b"correct=NULL; filter parse_ok=true before IRT fitting",
        b"unsupported_tasks": b"absent rows, not incorrect responses",
        b"native_labels": b"preserved in corresponding JSON archives; see model.label_mapping for binary collapse",
        b"item_source_sha256": hashlib.sha256(Path(D, "data/eval_items.jsonl").read_bytes()).hexdigest().encode(),
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    n_rows = 0
    cells = collections.Counter()
    native_counts = collections.Counter()
    finish_reasons = collections.Counter()
    input_at_limit = collections.Counter()
    cell_metrics = {}
    parse_failure_details = []
    archives = []
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for key in keys:
            seen = set()
            expected = {item_id for item_id, it in items.items() if it["task"] in supported_tasks(key)}
            for path in sorted(outdir.glob(f"{key}__*.json")):
                raw = path.read_bytes()
                doc = json.loads(raw)
                rows = []
                cell_key = f"{key}::{doc['benchmark']}::{doc['task']}"
                cell_metrics[cell_key] = doc.get("metrics")
                input_limit = doc.get("sampling", {}).get("truncate_prompt_tokens") or doc.get("sampling", {}).get("max_input_tokens")
                for result in doc["results"]:
                    item_id = result["item_id"]
                    if item_id not in expected or item_id in seen:
                        raise ValueError(f"Unexpected or duplicate row: {key}, {item_id}")
                    seen.add(item_id)
                    item = items[item_id]
                    if (doc["benchmark"], doc["task"], result["label_gt"]) != (item["benchmark"], item["task"], item["label"]):
                        raise ValueError(f"Gold-label or cell mismatch: {key}, {item_id}")
                    correct = int(result["pred_raw"] == item["label"]) if result["parse_ok"] else None
                    if not result["parse_ok"]:
                        parse_failure_details.append(dict(model_key=key, item_id=item_id,
                            finish_reasons=[call["finish_reason"] for call in result.get("calls", [])]))
                    rows.append(dict(model_key=key, item_id=item_id, benchmark=item["benchmark"],
                                     task=item["task"], correct=correct, p_positive=result["p_positive"],
                                     parse_ok=result["parse_ok"]))
                    cells[f"{key}::{item['benchmark']}::{item['task']}"] += 1
                    if isinstance(result["pred_native"], str):
                        native_counts[f"{key}::{item['task']}::{result['pred_native']}"] += 1
                    for call in result.get("calls", []):
                        finish_reasons[f"{key}::{call['finish_reason']}"] += 1
                        if input_limit and call["n_prompt_tokens"] >= input_limit:
                            input_at_limit[key] += 1
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                n_rows += len(rows)
                archives.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                     model_revision=doc["model"]["revision"], rows=len(rows)))
            if seen != expected:
                raise ValueError(f"Missing {len(expected-seen)} rows for {key}")
    check = pq.read_table(temporary)
    task_counts = collections.Counter(it["task"] for it in items.values())
    expected_rows = sum(sum(task_counts[task] for task in supported_tasks(key)) for key in keys)
    if check.num_rows != n_rows or n_rows != expected_rows:
        raise ValueError("Parquet row-count mismatch")
    if check.column("correct").null_count != sum(v["n_parse_errors"] for v in validation.values()):
        raise ValueError("Parse failures were not represented as null correctness")
    os.replace(temporary, destination)
    report = dict(parquet=str(destination), rows=n_rows, models=len(keys), archive_files=len(archives),
                  parse_failures=check.column("correct").null_count,
                  missing_probabilities=check.column("p_positive").null_count,
                  validation=validation, cells=dict(cells), native_counts=dict(native_counts), archives=archives,
                  cell_metrics=cell_metrics, finish_reasons=dict(finish_reasons),
                  calls_at_input_limit=dict(input_at_limit),
                  parse_failure_details=parse_failure_details,
                  implementation_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sorted(Path(D, "scripts").glob("*.py"))},
                  item_source_sha256=schema.metadata[b"item_source_sha256"].decode())
    report_path = destination.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("parquet", "rows", "models", "archive_files", "parse_failures", "missing_probabilities")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=f"{D}/outputs")
    parser.add_argument("--output", default=f"{D}/outputs/irt_long.parquet")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    keys = args.models or [m["key"] for m in MODELS if not m.get("blocked")]
    consolidate(args.outdir, args.output, keys)
