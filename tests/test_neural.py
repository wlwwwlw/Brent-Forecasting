import sys, json, unittest, csv
from statistics import mean
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neural_models import prepare, ForecastNet, fit_prepared, configure

ROOT = Path(__file__).resolve().parents[1]


class NeuralTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((ROOT / "config/neural.json").read_text())
        self.settings["epochs"] = 3
        self.prices = 100 + np.cumsum(np.random.default_rng(11).normal(size=100))

    def test_sequence_and_target_alignment(self):
        d = prepare(self.prices, 5)
        self.assertEqual(d["X"].shape, (75, 20, 1))
        self.assertEqual(d["last_training_origin"], 94)
        self.assertEqual(d["last_training_target"], 99)
        raw = d["X"].numpy() * d["x_scale"] + d["x_mean"]
        expected = self.prices[1:21] / self.prices[:20] - 1
        np.testing.assert_allclose(raw[0, :, 0], expected, atol=1e-8)
        expected = self.prices[75:95] / self.prices[74:94] - 1
        np.testing.assert_allclose(raw[-1, :, 0], expected, atol=1e-8)
        np.testing.assert_allclose(
            d["y"].numpy() * d["y_scale"] + d["y_mean"],
            self.prices[25:] - self.prices[20:95],
            atol=1e-6,
        )

    def test_scaler_excludes_current_prediction_window(self):
        p = self.prices.copy()
        p[-1] = 500
        d = prepare(p, 5)
        rs = p[1:] / p[:-1] - 1
        train = np.stack([rs[j - 20 : j] for j in range(20, 95)])
        self.assertAlmostEqual(d["x_mean"], train.mean())
        self.assertAlmostEqual(d["x_scale"], train.std())
        self.assertNotAlmostEqual(d["x_mean"], np.vstack([train, rs[-20:]]).mean())

    def test_all_architectures_have_correct_output_and_gradients(self):
        for c in self.settings["candidates"]:
            configure(7)
            model = ForecastNet(c["architecture"], c["hidden_size"])
            x = torch.randn(4, 20, 1)
            y = model(x)
            self.assertEqual(y.shape, (4,))
            y.square().mean().backward()
            self.assertTrue(
                all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            )

    def test_repeat_seed_is_reproducible(self):
        d = prepare(self.prices, 1)
        for c in self.settings["candidates"]:
            a, da = fit_prepared(d, c, self.settings, 42)
            b, db = fit_prepared(d, c, self.settings, 42)
            self.assertEqual(a, b)
            self.assertEqual(da["final_train_mse_standardized"], db["final_train_mse_standardized"])

    def test_constant_history_is_finite(self):
        d = prepare([100.0] * 100, 5)
        self.assertTrue(torch.isfinite(d["X"]).all())
        self.assertTrue(torch.isfinite(d["y"]).all())
        self.assertEqual(d["x_scale"], 1.0)
        self.assertEqual(d["y_scale"], 1.0)

    def test_invalid_history_is_rejected(self):
        for history in [[100.0] * 20, [100.0] * 99 + [float("nan")], [100.0] * 99 + [0.0]]:
            with self.assertRaises(ValueError):
                prepare(history, 5)

    def test_saved_neural_predictions_use_all_origins_and_seeds(self):
        out = ROOT / "results/validation"
        if not (out / "neural_run.json").exists():
            self.skipTest("Neural run not complete")
        rows = list(csv.DictReader((out / "neural_predictions.csv").read_text().splitlines()))
        baseline = [
            r
            for r in csv.DictReader((out / "baseline_predictions.csv").read_text().splitlines())
            if r["candidate"] == "naive"
        ]
        reference = {(int(r["origin_index"]), int(r["horizon"])): r for r in baseline}
        settings = json.loads((ROOT / "config/neural.json").read_text())
        self.assertEqual(
            len(rows), len(settings["candidates"]) * len(settings["seeds"]) * len(reference)
        )
        for c in settings["candidates"]:
            for seed in settings["seeds"]:
                subset = [r for r in rows if r["candidate"] == c["id"] and int(r["seed"]) == seed]
                keys = [(int(r["origin_index"]), int(r["horizon"])) for r in subset]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(set(keys), set(reference))
                for r, key in zip(subset, keys):
                    self.assertEqual(r["actual"], reference[key]["actual"])
                    self.assertEqual(r["target_date"], reference[key]["target_date"])
                    self.assertLessEqual(int(r["target_index"]), 207)
        entries = [
            json.loads(l) for l in (out / "neural_checkpoint.jsonl").read_text().splitlines() if l
        ]
        self.assertEqual(len(entries), len(rows))
        for r in entries:
            self.assertEqual(r["status"], "ok")
            d = r["diagnostics"]
            self.assertEqual(d["last_training_target"], r["origin_index"])
            self.assertEqual(d["last_training_origin"] + r["horizon"], r["origin_index"])
            self.assertEqual(d["training_n"], r["origin_index"] - r["horizon"] - 19)

    def test_aggregate_is_mean_of_seed_metrics_not_best_seed(self):
        out = ROOT / "results/validation"
        if not (out / "neural_run.json").exists():
            self.skipTest("Neural run not complete")
        seed_rows = list(csv.DictReader((out / "neural_seed_metrics.csv").read_text().splitlines()))
        aggregate = list(csv.DictReader((out / "neural_metrics.csv").read_text().splitlines()))
        for r in aggregate:
            subset = [
                s for s in seed_rows if all(r[k] == s[k] for k in ["candidate", "horizon", "fold"])
            ]
            self.assertEqual(len(subset), 3)
            self.assertAlmostEqual(float(r["RMSE_mean"]), mean(float(s["RMSE"]) for s in subset))

    def test_actual_saved_forecasts_can_be_reproduced(self):
        out = ROOT / "results/validation"
        if not (out / "neural_run.json").exists():
            self.skipTest("Neural run not complete")
        from experiment import load_rows

        settings = json.loads((ROOT / "config/neural.json").read_text())
        history = [r["close"] for r in load_rows()[:100]]
        data = prepare(history, 5)
        rows = list(csv.DictReader((out / "neural_predictions.csv").read_text().splitlines()))
        for c in settings["candidates"]:
            saved = next(
                r
                for r in rows
                if r["candidate"] == c["id"]
                and r["seed"] == "42"
                and r["horizon"] == "5"
                and r["origin_index"] == "99"
            )
            value, _ = fit_prepared(data, c, settings, 42)
            self.assertAlmostEqual(value, float(saved["predicted"]), places=7)


if __name__ == "__main__":
    unittest.main()
