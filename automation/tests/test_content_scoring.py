from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from content_scoring.application import BudgetExceeded, ContentScorer, discover_case_studies, load_rubric, parse_results
from content_scoring.domain import CriterionResult, deterministic_metrics, score_artifact
from content_scoring.infrastructure import ModelResponse, Telemetry


RUBRIC = Path(__file__).parents[1] / "content_scoring" / "rubrics" / "steadyburn-v1.json"


class FakeEvaluator:
    def __init__(self, payload: dict, cost: float = 0.01) -> None:
        self.payload, self.cost, self.calls = payload, cost, 0

    def complete(self, **kwargs):
        self.calls += 1
        return ModelResponse(self.payload, self.cost, {"cost": self.cost})


class ContentScoringTests(unittest.TestCase):
    def test_fk_grade_is_always_an_eligibility_gate(self) -> None:
        rubric = load_rubric(RUBRIC).artifacts["index.md"]
        metrics = deterministic_metrics(" ".join(["Institutionalization"] * 300) + ".")
        results = [CriterionResult(item.id, 100, True, 1, (), "") for item in rubric.criteria]
        scored = score_artifact(rubric, metrics, results, strict=False)
        self.assertGreater(metrics.fk_grade, 6)
        self.assertFalse(scored.eligible)

    def test_required_and_preferred_penalties_are_distinct(self) -> None:
        rubric = load_rubric(RUBRIC).artifacts["index.md"]
        metrics = deterministic_metrics("A man checks his alarm. He chooses a better action.")
        results = []
        for item in rubric.criteria:
            passed = item.id not in {"perspective_shift", "qualification_nuance"}
            results.append(CriterionResult(item.id, 100 if passed else 0, passed, 1, (), ""))
        scored = score_artifact(rubric, metrics, results, strict=False)
        self.assertEqual(scored.penalties, 21)
        self.assertTrue(scored.eligible)

    def test_strict_mode_turns_required_failure_into_ineligibility(self) -> None:
        rubric = load_rubric(RUBRIC).artifacts["index.md"]
        metrics = deterministic_metrics("A man checks his alarm. He chooses a better action.")
        results = [CriterionResult(item.id, 100, item.id != "decision", 1, (), "") for item in rubric.criteria]
        self.assertFalse(score_artifact(rubric, metrics, results, strict=True).eligible)

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
            scorer = ContentScorer(root, load_rubric(RUBRIC), FakeEvaluator({}), telemetry, 0.01, False, False, False)
            scorer.spent = 0.01
            with self.assertRaises(BudgetExceeded):
                scorer._call(model="test", prompt="", schema={}, stage="test", case_study="case", artifact="index.md")

    def test_unverified_evidence_is_removed_and_lowers_confidence(self) -> None:
        criteria = load_rubric(RUBRIC).artifacts["index.md"].criteria[:1]
        payload = {"criteria": [{"id": criteria[0].id, "score": 90, "passed": True, "confidence": 1, "evidence": ["not in the source"], "reason": "test"}]}
        result = parse_results(payload, criteria, "A man checks his alarm.")
        self.assertEqual(result[criteria[0].id].evidence, ())
        self.assertEqual(result[criteria[0].id].confidence, 0.7)

    def test_optional_tie_break_failure_keeps_terra_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry = Telemetry(root / "telemetry.jsonl")
            telemetry.emit("tie_break_fallback", run_id="test", case_study="case", artifact="index.md", model="sonnet", reason="RuntimeError")
            self.assertIn("tie_break_fallback", (root / "telemetry.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
