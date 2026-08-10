from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT.parent / "docs" / "model-comparisons"))
from generate_cost_charts import chart, discover_runs  # noqa: E402


class CostChartTests(unittest.TestCase):
    def test_local_baseline_is_zero_in_linear_and_log_charts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local" / "week"
            paid = root / "paid" / "week"
            local.mkdir(parents=True)
            paid.mkdir(parents=True)
            (local / "MODEL_COMPARISON.md").write_text("- Model: local/gemma\n", encoding="utf-8")
            (paid / "MODEL_COMPARISON.md").write_text("- Model: openai/gpt-test\n", encoding="utf-8")
            (paid / "OPENROUTER_USAGE.jsonl").write_text(json.dumps({"model": "openai/gpt-test", "usage": {"cost": 1.0}}) + "\n", encoding="utf-8")

            runs = discover_runs(root)

            self.assertEqual(next(run for run in runs if run.model == "local/gemma").total_cost, 0.0)
            self.assertIn("local baseline", chart(runs, logarithmic=False))
            self.assertIn("local baseline", chart(runs, logarithmic=True))


if __name__ == "__main__":
    unittest.main()
