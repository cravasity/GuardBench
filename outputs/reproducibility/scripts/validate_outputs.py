"""Check archive coverage and schema against canonical human-labelled items."""
import argparse
import collections
import json
import math
from pathlib import Path

import adapters
from models_manifest import MODELS
from runner import load_items, metrics


def supported_tasks(key):
    if adapters.pick(key):
        return adapters.pick(key).supports
    from hf_cls import SPECS
    return SPECS[key]["supports"]


def validate_model(key, outdir, limit=None, since=None, max_parse_error=0.0,
                   strict_parsing=True, require_probabilities=True):
    model = next(m for m in MODELS if m["key"] == key)
    native_adapter = adapters.pick(key)() if model["runner"] == "vllm_gen" else None
    expected = collections.defaultdict(dict)
    for item in load_items(limit, tasks=supported_tasks(key)):
        expected[(item["benchmark"], item["task"])][item["item_id"]] = item
    errors, n, bad, missing_p, bad_calls, missing_call_p = [], 0, 0, 0, 0, 0
    for (benchmark, task), gold in expected.items():
        path = Path(outdir) / f"{key}__{benchmark}__{task}.json"
        if not path.exists() or (since is not None and path.stat().st_mtime < since):
            errors.append(f"missing or stale: {path.name}")
            continue
        try:
            doc = json.loads(path.read_text())
            rows = doc["results"]
            assert doc["schema_version"] == "2.0"
            assert (doc["model"]["key"], doc["benchmark"], doc["task"]) == (key, benchmark, task)
            assert doc["model"]["revision"] and len(doc["model"]["revision"]) == 40
            if native_adapter is not None:
                assert doc["model"]["adapter_version"] == native_adapter.version, "stale adapter version"
            assert len(rows) == doc["n_items"] == len(gold)
            assert len({r["item_id"] for r in rows}) == len(rows)
            assert {r["item_id"] for r in rows} == set(gold)
            assert doc["metrics"] == metrics(rows)
            for row in rows:
                item = gold[row["item_id"]]
                assert row["label_gt"] == item["label"]
                assert row["label_gt_str"] == item["label_str"]
                assert type(row["parse_ok"]) is bool
                assert row["pred_raw"] in (None, 0, 1)
                assert row["parse_ok"] == (row["pred_raw"] is not None)
                assert row["pred"] == (row["pred_raw"] if row["parse_ok"] else 0)
                assert isinstance(row["calls"], list)
                assert row["n_calls"] == len(row["calls"]) == (4 if key.startswith("shieldgemma-") else 1)
                assert "pred_native" in row
                p = row["p_positive"]
                assert p is None or (math.isfinite(p) and 0 <= p <= 1)
                n += 1
                bad += not row["parse_ok"]
                missing_p += row["parse_ok"] and p is None
                for call in row["calls"]:
                    assert all(k in call for k in ("raw_output", "finish_reason", "label_native", "decision",
                                                   "prompt_sha1", "n_prompt_tokens", "n_output_tokens"))
                    assert all(k in call["decision"] for k in ("token_index", "token", "top_logprobs"))
                    bad_calls += call["pred"] is None
                    assert call["pred"] in (None, 0, 1)
                    if native_adapter is not None:
                        parsed = native_adapter.parse_call(call["call_id"], call["raw_output"], task)
                        assert call["pred"] == parsed.get("pred"), "prediction differs from raw output"
                        assert call["label_native"] == parsed.get("label_native"), "native label differs from raw output"
                    call_p = call["p_positive"]
                    assert call_p is None or (math.isfinite(call_p) and 0 <= call_p <= 1)
                    missing_call_p += call["pred"] is not None and call_p is None
                    if call["decision"]["token_index"] is not None:
                        assert len(call["decision"]["top_logprobs"]) >= 20
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid {path.name}: {type(exc).__name__} {exc}")
    if not n:
        errors.append("no validated items")
    if strict_parsing and n and bad / n > max_parse_error:
        errors.append(f"parse failures {bad}/{n}")
    if require_probabilities and missing_p:
        errors.append(f"parsed items without probabilities: {missing_p}/{n}")
    if require_probabilities and missing_call_p:
        errors.append(f"parsed calls without probabilities: {missing_call_p}")
    if strict_parsing and bad_calls:
        errors.append(f"unparsed calls: {bad_calls}")
    return dict(ok=not errors, n_items=n, n_parse_errors=bad, missing_probabilities=missing_p,
                unparsed_calls=bad_calls, missing_call_probabilities=missing_call_p, errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    keys = args.models or [m["key"] for m in MODELS
                           if not m.get("blocked") and (adapters.pick(m["key"]) or m["runner"] == "hf_cls")]
    report = {key: validate_model(key, args.outdir, args.limit) for key in keys}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(r["ok"] for r in report.values()) else 1)
