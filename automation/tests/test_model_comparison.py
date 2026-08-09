from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "case-studies"))
from generate_model_comparison import compile_comparison, dashboard_html, radar_svg  # noqa: E402


class ModelComparisonTests(unittest.TestCase):
    def test_compiles_every_dimension_and_cost_without_inventing_local_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            studies = workspace / "case-studies"
            rubric_directory = workspace / "content_scoring" / "rubrics"
            rubric_directory.mkdir(parents=True)
            shutil.copyfile(ROOT / "content_scoring" / "rubrics" / "steadyburn-v1.json", rubric_directory / "steadyburn-v1.json")
            rubric = json.loads((rubric_directory / "steadyburn-v1.json").read_text(encoding="utf-8"))
            self._write_study(studies, rubric, "openai/gpt-test", 0.50)
            self._write_study(studies, rubric, "local/gemma-test", None)

            data = compile_comparison(studies)

            self.assertEqual(len(data["models"]), 2)
            by_model = {item["model"]: item for item in data["models"]}
            paid, local = by_model["openai/gpt-test"], by_model["local/gemma-test"]
            self.assertEqual(paid["radar"]["cost_efficiency"], 100.0)
            self.assertIsNone(local["cost_usd"])
            self.assertIsNone(local["radar"]["cost_efficiency"])
            self.assertEqual(set(paid["criteria"]["index.md"]), {item["id"] for item in rubric["artifacts"]["index.md"]["criteria"]})
            self.assertIn("cdn.jsdelivr.net/npm/d3@7", dashboard_html(data))
            self.assertIn("<svg", radar_svg(data))
            self.assertIn("GPT Test", radar_svg(data))

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


if __name__ == "__main__":
    unittest.main()
