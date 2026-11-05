import sys
import unittest
import json
import hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_final import bootstrap_difference
from experiment import ROOT


class BootstrapTests(unittest.TestCase):
    def test_identical_forecasts_have_zero_difference(self):
        y = np.arange(20.0)
        self.assertEqual(bootstrap_difference(y, y + 1, y + 1, 5, 100), (0.0, 0.0))

    def test_known_constant_difference_and_pairing(self):
        y = np.arange(20.0)
        self.assertEqual(bootstrap_difference(y, y + 2, y + 1, 5, 100), (1.0, 1.0))

    def test_reproducible_and_direction_reverses(self):
        y = np.arange(20.0)
        p = y + np.sin(y)
        b = y + np.cos(y)
        lo, hi = bootstrap_difference(y, p, b, 5, 1000)
        reverse = bootstrap_difference(y, b, p, 5, 1000)
        self.assertEqual((lo, hi), bootstrap_difference(y, p, b, 5, 1000))
        np.testing.assert_allclose(reverse, (-hi, -lo), atol=1e-12)

    def test_reject_bad_alignment(self):
        with self.assertRaises(ValueError):
            bootstrap_difference([1, 2], [1], [1, 2], 1)


class FinalArtifactTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "results/test/predictions.csv").exists(), "Final test not yet run")
    def test_complete_aligned_results_and_no_future_targets(self):
        import pandas as pd

        df = pd.read_csv(ROOT / "results/test/predictions.csv")
        self.assertEqual(len(df), 552)
        self.assertFalse(df.duplicated(["family", "horizon", "seed", "origin_index"]).any())
        for _, group in df.groupby(["family", "horizon", "seed"]):
            self.assertEqual(sorted(group.origin_index), list(range(207, 253)))
            self.assertTrue((group.target_index == group.origin_index + group.horizon).all())
        for _, group in df.groupby(["horizon", "origin_index"]):
            self.assertEqual(group.actual.nunique(), 1)
            self.assertEqual(group.current.nunique(), 1)
        naive = df[df.family == "simple"]
        np.testing.assert_array_equal(naive.predicted, naive.current)
        diagnostics = json.loads((ROOT / "results/test/diagnostics.json").read_text())
        self.assertEqual(len(diagnostics), 552)
        for d in diagnostics:
            self.assertLessEqual(
                d.get("last_training_target", d["origin_index"]), d["origin_index"]
            )
        lock_path = ROOT / "config/final_lock.json"
        lock = json.loads(lock_path.read_text())
        for name, digest in lock["hashes"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        run = json.loads((ROOT / "results/test/run.json").read_text())
        self.assertEqual(run["lock_sha256"], hashlib.sha256(lock_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
