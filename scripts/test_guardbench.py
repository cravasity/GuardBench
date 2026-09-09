import math
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adapters import Call, DynaGuard, GPTOssSafeguard, GuardReasoner, Qwen3Guard, ShieldGemma
from runner import decision_probs, metrics
from drive import wait_for_memory
from consolidate_irt import consolidate


class GuardBenchTests(unittest.TestCase):
    def test_duplicate_decoded_tokens_are_retained(self):
        lp = {1: SimpleNamespace(decoded_token="Yes", logprob=math.log(0.6)),
              2: SimpleNamespace(decoded_token="Yes", logprob=math.log(0.2)),
              3: SimpleNamespace(decoded_token="No", logprob=math.log(0.2))}
        out = SimpleNamespace(logprobs=[lp], token_ids=[1], text="Yes")
        prob, table, index, token = decision_probs(out, Call("main", "", ["yes"], ["no"]))
        self.assertAlmostEqual(prob, 0.8)
        self.assertEqual(len(table), 3)
        self.assertEqual((index, token), (0, "Yes"))

    def test_missing_anchor_does_not_score_reasoning(self):
        out = SimpleNamespace(logprobs=[], token_ids=[], text="This may be unsafe")
        result = decision_probs(out, Call("main", "", ["unsafe"], ["safe"], anchor="Safety:"))
        self.assertIsNone(result[0])

    def test_unaligned_anchor_does_not_fall_back_to_reasoning(self):
        lp = {1: SimpleNamespace(decoded_token="unsafe", logprob=0.0)}
        out = SimpleNamespace(logprobs=[lp], token_ids=[1], text="unsafe reasoning; Safety:")
        result = decision_probs(out, Call("main", "", ["unsafe"], ["safe"], anchor="Safety:"))
        self.assertIsNone(result[0])

    def test_controversial_first_token_excludes_content(self):
        lp = {1: SimpleNamespace(decoded_token="Cont", logprob=math.log(0.3)),
              2: SimpleNamespace(decoded_token="Unsafe", logprob=math.log(0.1)),
              3: SimpleNamespace(decoded_token="Safe", logprob=math.log(0.4)),
              4: SimpleNamespace(decoded_token="Content", logprob=math.log(0.2))}
        out = SimpleNamespace(logprobs=[lp], token_ids=[1], text="Controversial")
        result = decision_probs(out, Call("main", "", ["unsafe", "=cont"], ["safe"]))
        self.assertAlmostEqual(result[0], 0.5)

    def test_qwen_preserves_controversial(self):
        parsed = Qwen3Guard().parse_call("main", "Safety: Controversial\nCategories: None", "prompt_harm")
        self.assertEqual(parsed["label_native"], "Controversial")
        self.assertEqual(parsed["pred"], 1)

    def test_dynaguard_rejects_labels_in_reasoning(self):
        guard = DynaGuard()
        self.assertIsNone(guard.parse_call("main", "We should decide PASS or FAIL", "prompt_harm")["pred"])
        self.assertEqual(guard.parse_call("main", "FAIL</answer>", "prompt_harm")["pred"], 1)

    def test_gptoss_final_channel_variants(self):
        guard = GPTOssSafeguard()
        for suffix in ["<|message|>1", " 1", " <|constrain|>1"]:
            text = "<|channel|>analysis<|message|>Maybe 0.<|end|><|start|>assistant<|channel|>final" + suffix
            self.assertEqual(guard.parse_call("main", text, "prompt_harm")["pred"], 1)
        self.assertIsNone(guard.parse_call("main", "<|channel|>analysis<|message|>Final answer 1", "prompt_harm")["pred"])

    def test_guardreasoner_uses_final_field(self):
        guard = GuardReasoner()
        text = "Response: unharmful\nAnswers:\nResponse: harmful"
        self.assertEqual(guard.parse_call("main", text, "response_harm")["pred"], 1)
        text = "Completion: refusal\nAnswers:\nCompletion: unharmful"
        self.assertIsNone(guard.parse_call("main", text, "response_refusal")["pred"])

    def test_guardreasoner_probability_uses_final_field(self):
        pieces = ["Response: ", "unharmful", "\nAnswers:\nResponse: ", "harmful"]
        lps = [{i: SimpleNamespace(decoded_token=piece, logprob=0.0)} for i, piece in enumerate(pieces)]
        lps[1] = {1: SimpleNamespace(decoded_token="unharmful", logprob=math.log(0.8)),
                  3: SimpleNamespace(decoded_token="harmful", logprob=math.log(0.2))}
        lps[3] = {1: SimpleNamespace(decoded_token="unharmful", logprob=math.log(0.1)),
                  3: SimpleNamespace(decoded_token="harmful", logprob=math.log(0.9))}
        out = SimpleNamespace(logprobs=lps, token_ids=list(range(4)), text="".join(pieces))
        call = Call("main", "", ["harmful"], ["unharmful"], GuardReasoner.decision_anchor("response_harm"))
        prob, _, index, _ = decision_probs(out, call)
        self.assertAlmostEqual(prob, 0.9)
        self.assertEqual(index, 3)

    def test_partial_shield_negative_is_not_a_valid_negative(self):
        guard = ShieldGemma()
        calls = [dict(call_id=str(i), pred=0 if i < 3 else None, p_positive=None) for i in range(4)]
        self.assertIsNone(guard.aggregate({}, calls)["pred"])
        calls[0]["pred"] = 1
        self.assertEqual(guard.aggregate({}, calls)["pred"], 1)

    def test_parse_failures_have_both_metric_views(self):
        rows = [dict(pred=0, label_gt=1, parse_ok=False, p_positive=None),
                dict(pred=1, label_gt=1, parse_ok=True, p_positive=0.8)]
        result = metrics(rows)
        self.assertEqual(result["metrics_conservative"]["accuracy"], 0.5)
        self.assertEqual(result["metrics_parsed_only"]["accuracy"], 1)

    @patch("drive.time.sleep")
    @patch("drive.subprocess.check_output", side_effect=["3000", "4", "4", "4"])
    def test_waits_for_all_reserved_gpus(self, query, sleep):
        wait_for_memory([0, 1])
        self.assertEqual(query.call_count, 4)
        sleep.assert_called_once_with(1)

    def test_parquet_keeps_failed_parse_missing(self):
        import pyarrow.parquet as pq
        items = [dict(item_id=str(i), benchmark="fixture", task="prompt_harm", label=label)
                 for i, label in enumerate([1, 0, 0])]
        rows = [dict(item_id=str(i), label_gt=label, pred_raw=pred, parse_ok=pred is not None,
                     p_positive=p, pred_native=native)
                for i, (label, pred, p, native) in enumerate([
                    (1, 1, 0.8, "Controversial"), (0, 1, 0.7, "Unsafe"), (0, None, None, None)])]
        report = dict(ok=True, n_parse_errors=1)
        with tempfile.TemporaryDirectory(dir="/raid/MLP/hgkim/guardbench/runtime") as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/eval_items.jsonl").write_text("fixture")
            (root / "guard__fixture__prompt_harm.json").write_text(json.dumps(dict(
                model=dict(revision="a" * 40), benchmark="fixture", task="prompt_harm", results=rows)))
            with patch("consolidate_irt.D", root), patch("consolidate_irt.load_items", return_value=items), \
                    patch("consolidate_irt.supported_tasks", return_value=["prompt_harm"]), \
                    patch("consolidate_irt.validate_model", return_value=report):
                audit = consolidate(root, root / "irt.parquet", ["guard"])
            table = pq.read_table(root / "irt.parquet")
            self.assertEqual(table.column_names, ["model_key", "item_id", "benchmark", "task",
                                                  "correct", "p_positive", "parse_ok"])
            self.assertEqual(table.column("correct").to_pylist(), [1, 0, None])
            self.assertEqual(audit["parse_failures"], 1)
            self.assertEqual(audit["native_counts"]["guard::prompt_harm::Controversial"], 1)

    def test_parquet_refuses_incomplete_archive(self):
        with patch("consolidate_irt.load_items", return_value=[]), \
                patch("consolidate_irt.validate_model", return_value=dict(ok=False, errors=["missing"])):
            with self.assertRaisesRegex(ValueError, "Incomplete or invalid archive"):
                consolidate("/nonexistent", "/nonexistent/irt.parquet", ["guard"])

    def test_parquet_refuses_duplicate_model_keys(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            consolidate("/nonexistent", "/nonexistent/irt.parquet", ["guard", "guard"])

    def test_parquet_refuses_duplicate_canonical_items(self):
        with patch("consolidate_irt.load_items", return_value=[dict(item_id="duplicate")] * 2):
            with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
                consolidate("/nonexistent", "/nonexistent/irt.parquet", ["guard"])


if __name__ == "__main__":
    unittest.main()
