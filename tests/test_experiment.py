import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment import metrics, forecast_simple, eligible_training_origins


class ExperimentTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        result = metrics([12.0, 8.0], [11.0, 10.0], [10.0, 10.0])
        self.assertAlmostEqual(result["MAE"], 1.5)
        self.assertAlmostEqual(result["RMSE"], (2.5) ** 0.5)
        self.assertAlmostEqual(result["sMAPE_pct"], (200 / 23 + 400 / 18) / 2)
        self.assertEqual(result["direction_accuracy_pct"], 50)
        self.assertEqual(result["direction_coverage_pct"], 50)

    def test_zero_denominator(self):
        self.assertEqual(metrics([0.0], [0.0], [0.0])["sMAPE_pct"], 0)

    def test_naive_has_no_direction(self):
        r = metrics([11.0, 9.0], [10.0, 10.0], [10.0, 10.0])
        self.assertEqual(r["direction_accuracy_pct"], 0)
        self.assertEqual(r["direction_coverage_pct"], 0)

    def test_bad_input(self):
        for a, p, c in [([], [], []), ([1.0], [1.0, 2.0], [1.0]), ([1.0], [float("nan")], [1.0])]:
            with self.assertRaises(ValueError):
                metrics(a, p, c)

    def test_training_labels_are_known(self):
        indices = eligible_training_origins(20, 5, 10)
        self.assertEqual(indices, list(range(9, 16)))
        self.assertTrue(all(i + 5 <= 20 for i in indices))

    def test_horizon_is_not_one_step_repeated_with_actual_future(self):
        history = [10.0, 12.0, 14.0]
        self.assertEqual(forecast_simple(history, 5, "drift"), 24)
        self.assertEqual(forecast_simple(history, 5, "naive"), 14)

    def test_seasonal_alignment(self):
        self.assertEqual(forecast_simple([1, 2, 3, 4, 5], 1, "seasonal_naive", 5), 1)
        self.assertEqual(forecast_simple([1, 2, 3, 4, 5], 5, "seasonal_naive", 5), 5)
        self.assertEqual(forecast_simple([1, 2, 3, 4, 5], 6, "seasonal_naive", 5), 1)

    def test_future_mutation_does_not_change_prediction(self):
        a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        b = a[:3] + [999.0, -999.0, 500.0, 0.0]
        for model, window in [
            ("naive", None),
            ("drift", None),
            ("rolling_mean", 3),
            ("rolling_median", 3),
            ("seasonal_naive", 3),
        ]:
            self.assertEqual(
                forecast_simple(a[:3], 5, model, window), forecast_simple(b[:3], 5, model, window)
            )


if __name__ == "__main__":
    unittest.main()
