"""Послепроверочный разбор коэффициентов выбранных линейных моделей; без смены сетки."""

import csv
import json
import numpy as np
from experiment import ROOT, load_rows, write_csv
from ml_models import prepare, estimator, FEATURE_NAMES


def main():
    config = json.loads((ROOT / "config/experiment.json").read_text())
    grid = json.loads((ROOT / "config/ml.json").read_text())
    run = json.loads((ROOT / "results/validation/ml_run.json").read_text())
    prices = [r["close"] for r in load_rows()[: config["test_origin_start"] + 1]]
    saved = list(
        csv.DictReader((ROOT / "results/validation/ml_predictions.csv").read_text().splitlines())
    )
    ref = {
        (r["candidate"], int(r["horizon"]), int(r["origin_index"])): float(r["predicted"])
        for r in saved
    }
    records = []
    for hs, cid in run["winners"].items():
        h = int(hs)
        candidate = next(c for c in grid["candidates"] if c["id"] == cid)
        if candidate["method"] not in ["ridge", "elastic"]:
            continue
        for fold in config["validation_folds"]:
            for t in range(fold["origin_start"], fold["origin_end"] + 1):
                data = prepare(prices[: t + 1], h)
                model = estimator(candidate, grid["seed"])
                model.fit(data["X"], data["y"])
                predicted = float(model.predict(data["current"])[0]) + data["price"]
                if not np.isclose(predicted, ref[cid, h, t], atol=1e-9, rtol=1e-10):
                    raise ValueError("Winner refit disagrees with saved forecast")
                coef = model.regressor_.named_steps["model"].coef_
                records.append(
                    dict(
                        horizon=h,
                        origin_index=t,
                        nonzero_coefficients=int((abs(coef) > 1e-12).sum()),
                        **dict(zip(FEATURE_NAMES, coef)),
                    )
                )
    write_csv(ROOT / "results/validation/ml_winner_coefficients.csv", records)
    print("Winner coefficients and forecast reproducibility checked.")


if __name__ == "__main__":
    main()
