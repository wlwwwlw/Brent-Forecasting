import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment import ROOT
from check_project import check_links, check_numbers, check_report_writers
import lock_finalists


class ProjectTests(unittest.TestCase):
    def test_project_links_numbers_and_report_destinations(self):
        check_links()
        check_numbers()
        check_report_writers()

    def test_wrong_readme_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "docs").mkdir()
            (root / "results/test").mkdir(parents=True)
            shutil.copy2(
                ROOT / "results/test/primary_metrics.csv", root / "results/test/primary_metrics.csv"
            )
            shutil.copy2(ROOT / "docs/analysis.md", root / "docs/analysis.md")
            text = (ROOT / "README.md").read_text(encoding="utf-8").replace("2.8989", "9.9999")
            (root / "README.md").write_text(text, encoding="utf-8")
            with self.assertRaises(AssertionError):
                check_numbers(root)

    def test_existing_final_lock_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config").mkdir()
            lock = root / "config/final_lock.json"
            before = b'{"original":true}\n'
            lock.write_bytes(before)
            with patch.object(lock_finalists, "ROOT", root):
                lock_finalists.main()
            self.assertEqual(lock.read_bytes(), before)
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
