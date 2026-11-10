from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check-content-png.sh"


class ContentPngGuardTests(unittest.TestCase):
    def test_fails_and_reports_png_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            offending = root / "content" / "articles" / "image.png"
            offending.parent.mkdir(parents=True)
            offending.touch()

            result = subprocess.run([str(CHECKER), str(root)], capture_output=True, text=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("content/articles/image.png", result.stderr)

    def test_succeeds_when_no_png_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "content" / "articles").mkdir(parents=True)

            result = subprocess.run([str(CHECKER), str(temp)], capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
