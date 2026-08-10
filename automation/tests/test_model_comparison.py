from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT.parent / "docs" / "model-comparisons"))
from generate_model_comparison import READABILITY_METRICS, compile_comparison, dashboard_html, radar_svg, radial_fraction  # noqa: E402


class ModelComparisonTests(unittest.TestCase):
    def test_compiles_every_dimension_and_cost_without_inventing_local_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            studies = workspace / "model-comparisons"
            rubric_directory = workspace / "content_scoring" / "rubrics"
            rubric_directory.mkdir(parents=True)
            shutil.copyfile(ROOT / "content_scoring" / "rubrics" / "steadyburn-v1.json", rubric_directory / "steadyburn-v1.json")
            rubric = json.loads((rubric_directory / "steadyburn-v1.json").read_text(encoding="utf-8"))
            self._write_study(studies, rubric, "openai/gpt-test", 0.50)
            self._write_study(studies, rubric, "local/gemma-test", None)
            self._write_readability_report(studies)

            data = compile_comparison(studies)

            self.assertEqual(len(data["models"]), 2)
            by_model = {item["model"]: item for item in data["models"]}
            paid, local = by_model["openai/gpt-test"], by_model["local/gemma-test"]
            self.assertEqual(paid["radar_raw"]["readability"]["cost_burden"], 100.0)
            self.assertEqual(local["cost_usd"], 0.0)
            self.assertEqual(local["radar_raw"]["readability"]["cost_burden"], 0.0)
            self.assertEqual(local["radar"]["readability"]["cost_burden"], 15.0)
            self.assertEqual(set(paid["criteria"]["index.md"]), {item["id"] for item in rubric["artifacts"]["index.md"]["criteria"]})
            self.assertEqual(set(paid["radar"]["readability"]), {metric for metric, _label in READABILITY_METRICS} | {"cost_burden"})
            self.assertEqual(set(paid["readability"]["bundle total"]), {metric for metric, _label in READABILITY_METRICS})
            self.assertEqual(len(data["axis_sets"]["quality"]), sum(len(item["criteria"]) for item in rubric["artifacts"].values()) + 1)
            self.assertIn("letter_simplicity", {axis["id"] for axis in data["axis_sets"]["overview"]})
            self.assertIn("instructions_economy", {axis["id"] for axis in data["axis_sets"]["overview"]})
            self.assertIn("cdn.jsdelivr.net/npm/d3@7", dashboard_html(data))
            self.assertIn("Math.log1p", dashboard_html(data))
            self.assertIn("<svg", radar_svg(data))
            self.assertIn("within-dimension rank", radar_svg(data))
            self.assertIn("GPT Test", radar_svg(data))

    def test_logarithmic_radar_scale_is_zero_safe_and_expands_low_values(self) -> None:
        self.assertEqual(radial_fraction(0), 0)
        self.assertEqual(radial_fraction(100), 1)
        self.assertGreater(radial_fraction(10), 0.1)

    def _write_study(self, root: Path, rubric: dict, model: str, cost: float | None) -> None:
        study = root / model.replace("/", "-") / "week"
        study.mkdir(parents=True)
        (study / "MODEL_COMPARISON.md").write_text(f"- Model: {model}\n", encoding="utf-8")
        artifacts = {}
        for artifact, artifact_rubric in rubric["artifacts"].items():
            artifacts[artifact] = {
                "metrics": {"fk_grade": 7.0},
                "criteria": [{"id": item["id"], "score": 80.0} for item in artifact_rubric["criteria"]],
            }
        (study / "CONTENT_SCORE.json").write_text(json.dumps({"model": model, "content_score": 80.0, "confidence": 0.9, "artifacts": artifacts}), encoding="utf-8")
        if cost is not None:
            (study / "OPENROUTER_USAGE.jsonl").write_text(json.dumps({"usage": {"cost": cost}}) + "\n", encoding="utf-8")

    def _write_readability_report(self, root: Path) -> None:
        records = []
        for index, comparison in enumerate(sorted(root.rglob("MODEL_COMPARISON.md"))):
            case_study = comparison.parent.relative_to(root).as_posix()
            records.append({
                "case_study": case_study,
                "document": "bundle total",
                **{metric: float(index + 1) for metric, _label in READABILITY_METRICS},
            })
        (root / "readability-report.json").write_text(json.dumps({"records": records}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
