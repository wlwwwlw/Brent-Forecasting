import sys
import csv
import json
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ml_models import features, prepare, estimator, fit_predict


class MLTests(unittest.TestCase):
    def test_features_on_known_growth(self):
        p = 100 * 1.01 ** np.arange(30)
        x = features(p)
        self.assertEqual(len(x), 14)
        np.testing.assert_allclose(x[:5], 0.01, atol=1e-12)
        np.testing.assert_allclose(x[5:8], [1.01**w - 1 for w in [5, 10, 20]], atol=1e-12)
        np.testing.assert_allclose(x[8:11], 0, atol=1e-12)

    def test_labels_only_use_known_prices(self):
        p = np.arange(100.0, 200.0)
        data = prepare(p, 5)
        self.assertEqual(data["X"].shape, (75, 14))
        self.assertEqual(data["origins"][0], 20)
        self.assertEqual(data["origins"][-1], 94)
        self.assertEqual(data["last_target"], 99)
        np.testing.assert_allclose(data["y"], 5)
        np.testing.assert_allclose(data["X"][-1], features(p[:95]))
        np.testing.assert_allclose(data["current"][0], features(p))

    def test_scalers_fit_on_training_examples_only(self):
        p = 100 + np.cumsum(np.random.default_rng(2).normal(size=100))
        p[-1] = 500
        data = prepare(p, 5)
        model = estimator({"method": "ridge", "alpha": 10.0})
        model.fit(data["X"], data["y"])
        xs = model.regressor_.named_steps["scale"]
        np.testing.assert_allclose(xs.mean_, data["X"].mean(axis=0))
        self.assertEqual(xs.n_samples_seen_, 75)
        self.assertFalse(
            np.allclose(xs.mean_, np.vstack([data["X"], data["current"]]).mean(axis=0))
        )
        self.assertAlmostEqual(model.transformer_.mean_[0], data["y"].mean())

    def test_change_forecast_restores_price_level(self):
        for method in [
            {"method": "ridge", "alpha": 10.0},
            {"method": "forest", "n_estimators": 10, "max_depth": 2, "min_samples_leaf": 5},
        ]:
            pred, _ = fit_predict(list(range(100, 200)), 5, method)
            self.assertAlmostEqual(pred, 204.0, places=7)

    def test_no_future_data_in_training_examples(self):
        a = np.arange(100.0, 230.0)
        b = a.copy()
        b[100:] = 99999
        da, db = prepare(a[:100], 5), prepare(b[:100], 5)
        for k in ["X", "y", "current"]:
            np.testing.assert_array_equal(da[k], db[k])

    def test_invalid_feature_values(self):
        for values in [[100] * 20, [100] * 20 + [0], [100] * 20 + [float("nan")]]:
            with self.assertRaises(ValueError):
                features(values)

    def test_saved_ml_forecasts_and_scalers_respect_protocol(self):
        root = Path(__file__).resolve().parents[1]
        out = root / "results/validation"
        if not (out / "ml_predictions.csv").exists():
            self.skipTest("Run ML first")
        rows = list(csv.DictReader((out / "ml_predictions.csv").read_text().splitlines()))
        naive = [
            r
            for r in csv.DictReader((out / "baseline_predictions.csv").read_text().splitlines())
            if r["candidate"] == "naive"
        ]
        ref = {(int(r["origin_index"]), int(r["horizon"])): r for r in naive}
        grid = json.loads((root / "config/ml.json").read_text())
        self.assertEqual(len(rows), len(grid["candidates"]) * len(ref))
        for candidate in grid["candidates"]:
            sub = [r for r in rows if r["candidate"] == candidate["id"]]
            keys = [(int(r["origin_index"]), int(r["horizon"])) for r in sub]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(set(keys), set(ref))
            for r, key in zip(sub, keys):
                for field in ["actual", "current", "target_date"]:
                    self.assertEqual(r[field], ref[key][field])
                self.assertLessEqual(int(r["target_index"]), 207)
        fits = json.loads((out / "ml_fit_diagnostics.json").read_text())
        self.assertEqual(len(fits), len(rows))
        for fit in fits:
            self.assertEqual(fit["last_training_target"], fit["origin_index"])
            self.assertEqual(fit["last_training_origin"] + fit["horizon"], fit["origin_index"])
            self.assertEqual(fit["scaler_n"], fit["training_n"])
            self.assertEqual(fit["training_n"], fit["origin_index"] - fit["horizon"] - 19)


if __name__ == "__main__":
    unittest.main()
