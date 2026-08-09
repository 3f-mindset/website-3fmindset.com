from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from content_scoring.application import EVIDENCE_MODEL, JUDGE_MODEL, TIE_BREAK_MODEL, BudgetExceeded, ContentScorer, discover_case_studies, load_rubric, parse_results
from content_scoring.domain import CriterionResult, deterministic_metrics, score_artifact
from content_scoring.infrastructure import ModelResponse, OpenRouterEvaluator, Telemetry


RUBRIC = Path(__file__).parents[1] / "content_scoring" / "rubrics" / "steadyburn-v1.json"


class FakeEvaluator:
    def __init__(self, payload: dict, cost: float = 0.01) -> None:
        self.payload, self.cost, self.calls = payload, cost, 0

    def complete(self, **kwargs):
        self.calls += 1
        return ModelResponse(self.payload, self.cost, {"cost": self.cost})


class ContentScoringTests(unittest.TestCase):
    def _evaluator_with_responses(self, responses: list[object], attempts: int = 4) -> tuple[OpenRouterEvaluator, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        telemetry_path = Path(temporary.name) / "telemetry.jsonl"
        client = MagicMock()
        client.post.side_effect = responses
        context = MagicMock()
        context.__enter__.return_value = client
        context.__exit__.return_value = False
        self.client_patch = patch("content_scoring.infrastructure.httpx.Client", return_value=context)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        return OpenRouterEvaluator("test-key", Telemetry(telemetry_path), attempts=attempts), telemetry_path

    @staticmethod
    def _response(content: str, finish_reason: str = "stop") -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}, "finish_reason": finish_reason}], "usage": {"cost": 0.01}},
            request=httpx.Request("POST", "https://example.test/evaluator"),
        )

    def test_transport_protocol_errors_are_retried(self) -> None:
        evaluator, telemetry_path = self._evaluator_with_responses([
            httpx.RemoteProtocolError("connection reset"),
            self._response('{"criteria": []}'),
        ], attempts=2)
        result = evaluator.complete(model="test", prompt="prompt", schema={}, run_id="run", stage="judge", case_study="case", artifact="index.md")
        events = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result.payload, {"criteria": []})
        self.assertTrue(any(event["event"] == "retry" and event["reason"] == "RemoteProtocolError" for event in events))
        self.assertEqual(next(event for event in events if event["event"] == "model_completed")["finish_reason"], "stop")

    def test_malformed_json_uses_the_full_retry_budget(self) -> None:
        evaluator, telemetry_path = self._evaluator_with_responses([
            self._response("not json"),
            self._response("still not json"),
            self._response("also not json"),
            self._response('{"criteria": []}'),
        ])
        result = evaluator.complete(model="test", prompt="prompt", schema={}, run_id="run", stage="judge", case_study="case", artifact="index.md")
        events = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
        retries = [event for event in events if event["event"] == "retry" and event["reason"] == "JSONDecodeError"]
        self.assertEqual(result.payload, {"criteria": []})
        self.assertEqual([event["attempt"] for event in retries], [1, 2, 3])

    def test_evaluator_stack_uses_only_deepseek_models(self) -> None:
        self.assertEqual(EVIDENCE_MODEL, "deepseek/deepseek-v4-flash")
        self.assertEqual(JUDGE_MODEL, "deepseek/deepseek-v4-pro")
        self.assertEqual(TIE_BREAK_MODEL, "deepseek/deepseek-v4-pro")

    def test_fk_grade_is_retained_as_a_measurement_only(self) -> None:
        rubric = load_rubric(RUBRIC).artifacts["index.md"]
        metrics = deterministic_metrics(" ".join(["Institutionalization"] * 300) + ".")
        results = [CriterionResult(item.id, 100, True, 1, (), "") for item in rubric.criteria]
        scored = score_artifact(rubric, metrics, results)
        self.assertGreater(metrics.fk_grade, 6)
        self.assertEqual(scored.final_score, 100)
        self.assertNotIn("eligible", scored.as_dict())

    def test_required_and_preferred_penalties_are_distinct(self) -> None:
        rubric = load_rubric(RUBRIC).artifacts["index.md"]
        metrics = deterministic_metrics("A man checks his alarm. He chooses a better action.")
        results = []
        for item in rubric.criteria:
            passed = item.id not in {"perspective_shift", "qualification_nuance"}
            results.append(CriterionResult(item.id, 100 if passed else 0, passed, 1, (), ""))
        scored = score_artifact(rubric, metrics, results)
        self.assertEqual(scored.penalties, 21)

    def test_discovery_requires_both_primary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = root / "model" / "week"
            full.mkdir(parents=True)
            (full / "MODEL_COMPARISON.md").write_text("- Model: test/model\n", encoding="utf-8")
            (full / "index.md").write_text("# Title\n", encoding="utf-8")
            self.assertEqual(discover_case_studies(root), [])
            (full / "INSTRUCTIONS.md").write_text("# Instructions\n", encoding="utf-8")
            studies = discover_case_studies(root)
            self.assertEqual(len(studies), 1)
            self.assertEqual(studies[0].model, "test/model")

    def test_budget_stops_before_new_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry = Telemetry(root / "telemetry.jsonl")
            scorer = ContentScorer(root, load_rubric(RUBRIC), FakeEvaluator({}), telemetry, 0.01, False, False)
            scorer.spent = 0.01
            with self.assertRaises(BudgetExceeded):
                scorer._call(model="test", prompt="", schema={}, stage="test", case_study="case", artifact="index.md")

    def test_unverified_evidence_is_removed_and_lowers_confidence(self) -> None:
        criteria = load_rubric(RUBRIC).artifacts["index.md"].criteria[:1]
        payload = {"criteria": [{"id": criteria[0].id, "score": 90, "passed": True, "confidence": 1, "evidence": ["not in the source"], "reason": "test"}]}
        result = parse_results(payload, criteria, "A man checks his alarm.")
        self.assertEqual(result[criteria[0].id].evidence, ())
        self.assertEqual(result[criteria[0].id].confidence, 0.7)

    def test_optional_tie_break_failure_keeps_deepseek_judge_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry = Telemetry(root / "telemetry.jsonl")
            telemetry.emit("tie_break_fallback", run_id="test", case_study="case", artifact="index.md", model="deepseek/deepseek-v4-pro", reason="RuntimeError")
            self.assertIn("tie_break_fallback", (root / "telemetry.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
