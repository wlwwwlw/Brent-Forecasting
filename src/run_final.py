"""Evaluate locked family winners on untouched rolling test origins."""

import hashlib
import json
import time
from experiment import ROOT, load_rows, write_csv
from statistical_models import fit_forecast
from ml_models import prepare as prepare_ml, fit_prepared as fit_ml
from neural_models import prepare as prepare_nn, fit_prepared as fit_nn
from integrity import verify_files


def main():
    lock_path = ROOT / "config/final_lock.json"
    lock = json.loads(lock_path.read_text())
    verify_files(lock["hashes"])
    settings = json.loads((ROOT / "config/experiment.json").read_text())
    grids = {
        f: json.loads((ROOT / f"config/{f}.json").read_text())
        for f in ["statistical", "ml", "neural"]
    }
    candidates = {f: {c["id"]: c for c in grid["candidates"]} for f, grid in grids.items()}
    rows = load_rows()
    prices = [r["close"] for r in rows]
    output = ROOT / "results/test"
    output.mkdir(parents=True, exist_ok=True)
    records, diagnostics = [], []
    for t in range(settings["test_origin_start"], settings["test_origin_end"] + 1):
        history = prices[: t + 1]
        for h in settings["horizons"]:
            for family in ["simple", "statistical", "ml", "neural"]:
                cid = lock["winners"][family][str(h)]
                seeds = lock["neural_seeds"] if family == "neural" else [42]
                for seed in seeds:
                    start = time.perf_counter()
                    diag = {}
                    if family == "simple":
                        if cid != "naive":
                            raise ValueError("Unexpected simple winner")
                        prediction = history[-1]
                    elif family == "statistical":
                        trajectory, diag = fit_forecast(history, h, candidates[family][cid])
                        prediction = trajectory[-1]
                    elif family == "ml":
                        prediction, diag = fit_ml(
                            prepare_ml(history, h), candidates[family][cid], seed
                        )
                    else:
                        prediction, diag = fit_nn(
                            prepare_nn(history, h, grids[family]["lookback"]),
                            candidates[family][cid],
                            grids[family],
                            seed,
                        )
                    if diag.get("last_training_target", t) > t:
                        raise AssertionError("Future target used")
                    seconds = time.perf_counter() - start
                    key = dict(family=family, candidate=cid, seed=seed, horizon=h, origin_index=t)
                    records.append(
                        dict(
                            **key,
                            origin_date=rows[t]["timestamp_utc"][:10],
                            target_index=t + h,
                            target_date=rows[t + h]["timestamp_utc"][:10],
                            current=prices[t],
                            actual=prices[t + h],
                            predicted=prediction,
                        )
                    )
                    diagnostics.append(dict(**key, fit_forecast_seconds=seconds, **diag))
        if (t - settings["test_origin_start"]) % 10 == 0:
            print(f"Completed origin {t}", flush=True)
    expected = 46 * 2 * 6
    if len(records) != expected:
        raise AssertionError("Incomplete test")
    write_csv(output / "predictions.csv", records)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (output / "run.json").write_text(
        json.dumps(
            dict(
                lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                forecasts=len(records),
                failures=0,
                total_fit_forecast_seconds=sum(d["fit_forecast_seconds"] for d in diagnostics),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} predictions; no failures.", flush=True)


if __name__ == "__main__":
    main()
