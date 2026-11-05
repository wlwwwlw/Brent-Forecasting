import sys
import csv
import json
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from statistical_models import fit_forecast


class StatisticalTests(unittest.TestCase):
    def test_ar_forecast_recursion_matches_known_process(self):
        history = [25.0]
        for i in range(49):
            history.append(3 + 0.8 * history[-1])
        predicted, _ = fit_forecast(history, 5, {"method": "ar", "lags": 1})
        x = history[-1]
        for p in predicted:
            x = 3 + 0.8 * x
            self.assertAlmostEqual(p, x, places=6)

    def test_random_walk_arima_is_naive(self):
        history = (100 + np.cumsum(np.random.default_rng(42).normal(size=80))).tolist()
        predicted, _ = fit_forecast(
            history, 5, {"method": "arima", "order": [0, 1, 0], "trend": "n"}
        )
        np.testing.assert_allclose(predicted, [history[-1]] * 5, atol=1e-7)

    def test_ses_flat_forecast(self):
        history = (100 + np.sin(np.arange(60) / 3)).tolist()
        predicted, _ = fit_forecast(history, 5, {"method": "ses"})
        np.testing.assert_allclose(predicted, [predicted[0]] * 5)

    def test_holt_recovers_linear_trend(self):
        predicted, _ = fit_forecast(list(range(100, 160)), 5, {"method": "holt"})
        np.testing.assert_allclose(predicted, list(range(160, 165)), atol=0.02)

    def test_forecast_scales_back_to_dollars(self):
        history = (100 + np.cumsum(np.random.default_rng(7).normal(size=70))).tolist()
        for candidate in [{"method": "ar", "lags": 5}, {"method": "holt_damped"}]:
            a, _ = fit_forecast(history, 5, candidate)
            b, _ = fit_forecast([v * 10 for v in history], 5, candidate)
            np.testing.assert_allclose(b, np.array(a) * 10, rtol=1e-6, atol=1e-6)

    def test_rejects_nonfinite_data(self):
        with self.assertRaises(ValueError):
            fit_forecast([100.0] * 30 + [float("nan")], 5, {"method": "ses"})

    def test_saved_results_cover_all_common_dates(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "results/validation"
        path = output / "statistical_predictions.csv"
        if not path.exists():
            self.skipTest("Run statistics first")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        baseline = list(
            csv.DictReader(
                (output / "baseline_predictions.csv").read_text(encoding="utf-8").splitlines()
            )
        )
        reference = {
            (int(r["origin_index"]), int(r["horizon"])): r
            for r in baseline
            if r["candidate"] == "naive"
        }
        candidates = json.loads((root / "config/statistical.json").read_text())["candidates"]
        self.assertEqual(len(rows), len(candidates) * len(reference))
        for candidate in candidates:
            subset = [r for r in rows if r["candidate"] == candidate["id"]]
            keys = [(int(r["origin_index"]), int(r["horizon"])) for r in subset]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(set(keys), set(reference))
            for r, key in zip(subset, keys):
                self.assertEqual(r["actual"], reference[key]["actual"])
                self.assertEqual(r["current"], reference[key]["current"])
                self.assertEqual(r["target_date"], reference[key]["target_date"])
                self.assertLessEqual(int(r["target_index"]), 207)


if __name__ == "__main__":
    unittest.main()
