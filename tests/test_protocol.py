"""Проверки реальных границ эксперимента и сопоставимости сохранённых прогнозов."""

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_validation_outcomes_available_before_test_forecast(self):
        config = json.loads((ROOT / "config/experiment.json").read_text())
        manifest = list(
            csv.DictReader(
                (ROOT / "config/split_manifest.csv").read_text(encoding="utf-8").splitlines()
            )
        )
        validation = [r for r in manifest if r["split"] != "test"]
        test = [r for r in manifest if r["split"] == "test"]
        self.assertLessEqual(
            max(int(r["target_index"]) for r in validation),
            min(int(r["origin_index"]) for r in test),
        )
        self.assertTrue(
            set(int(r["target_index"]) for r in validation).isdisjoint(
                int(r["target_index"]) for r in test
            )
        )
        for r in manifest:
            self.assertEqual(int(r["target_index"]) - int(r["origin_index"]), int(r["horizon"]))
            self.assertLess(int(r["target_index"]), config["n_active"])

    def test_saved_predictions_have_identical_origins(self):
        path = ROOT / "results/validation/baseline_predictions.csv"
        if not path.exists():
            self.skipTest("Run baselines first")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        origins = defaultdict(set)
        for r in rows:
            origins[(r["candidate"], r["horizon"])].add(int(r["origin_index"]))
            self.assertLessEqual(int(r["target_index"]), 207)
        self.assertEqual(len(rows), 1872)
        self.assertEqual(len(origins), 18)
        reference = next(iter(origins.values()))
        self.assertEqual(len(reference), 104)
        self.assertTrue(all(s == reference for s in origins.values()))


if __name__ == "__main__":
    unittest.main()
