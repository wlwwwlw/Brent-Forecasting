"""ML-валидация: отдельные прямые прогнозы h=1 и h=5, без финального теста."""

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from experiment import ROOT, load_rows, metrics, write_csv
from ml_models import prepare, fit_prepared, FEATURE_NAMES


def main():
    settings = json.loads((ROOT / "config/experiment.json").read_text(encoding="utf-8"))
    grid_path = ROOT / "config/ml.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    source = ROOT / "data/processed/brent_active_provisional.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != settings["processed_sha256"]:
        raise ValueError("Data changed")
    rows = load_rows()[: settings["test_origin_start"] + 1]
    close = [r["close"] for r in rows]
    output = ROOT / "results/validation"
    output.mkdir(exist_ok=True, parents=True)
    expected = sum(f["origin_end"] - f["origin_start"] + 1 for f in settings["validation_folds"])
    total_start = time.perf_counter()
    cache = {}
    prep_start = time.perf_counter()
    for fold in settings["validation_folds"]:
        for t in range(fold["origin_start"], fold["origin_end"] + 1):
            for h in settings["horizons"]:
                if t + h > settings["test_origin_start"]:
                    raise ValueError("Validation crosses test")
                cache[t, h] = prepare(close[: t + 1], h)
    feature_seconds = time.perf_counter() - prep_start
    records, fits, failures = [], [], []
    for candidate in grid["candidates"]:
        print("Fitting", candidate["id"], flush=True)
        for h in settings["horizons"]:
            for fold in settings["validation_folds"]:
                for t in range(fold["origin_start"], fold["origin_end"] + 1):
                    start = time.perf_counter()
                    try:
                        pred, diag = fit_prepared(cache[t, h], candidate, grid["seed"])
                    except Exception as exc:
                        failures.append(
                            dict(
                                candidate=candidate["id"],
                                horizon=h,
                                origin_index=t,
                                error=repr(exc),
                            )
                        )
                        continue
                    elapsed = time.perf_counter() - start
                    fits.append(
                        dict(
                            candidate=candidate["id"],
                            horizon=h,
                            origin_index=t,
                            fit_forecast_seconds=elapsed,
                            **diag,
                        )
                    )
                    records.append(
                        dict(
                            candidate=candidate["id"],
                            family="ml",
                            method=candidate["method"],
                            fold=fold["id"],
                            horizon=h,
                            origin_index=t,
                            origin_date=rows[t]["timestamp_utc"][:10],
                            target_index=t + h,
                            target_date=rows[t + h]["timestamp_utc"][:10],
                            current=close[t],
                            actual=close[t + h],
                            predicted=pred,
                        )
                    )
        (output / "ml_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    summaries = []
    for candidate in grid["candidates"]:
        for h in settings["horizons"]:
            pool = [r for r in records if r["candidate"] == candidate["id"] and r["horizon"] == h]
            if len(pool) != expected:
                continue
            for fold_id in ["all"] + [f["id"] for f in settings["validation_folds"]]:
                subset = [r for r in pool if fold_id == "all" or r["fold"] == fold_id]
                score = metrics(
                    [r["actual"] for r in subset],
                    [r["predicted"] for r in subset],
                    [r["current"] for r in subset],
                )
                summaries.append(
                    dict(
                        candidate=candidate["id"],
                        method=candidate["method"],
                        horizon=h,
                        fold=fold_id,
                        **score,
                    )
                )
    key = lambda r: (r["RMSE"], r["MAE"], r["candidate"])
    winners = {
        str(h): min([r for r in summaries if r["horizon"] == h and r["fold"] == "all"], key=key)[
            "candidate"
        ]
        for h in settings["horizons"]
    }
    method_winners = {
        str(h): {
            m: min(
                [
                    r
                    for r in summaries
                    if r["horizon"] == h and r["fold"] == "all" and r["method"] == m
                ],
                key=key,
            )["candidate"]
            for m in sorted(set(r["method"] for r in summaries))
        }
        for h in settings["horizons"]
    }
    write_csv(output / "ml_predictions.csv", records)
    write_csv(output / "ml_metrics.csv", summaries)
    (output / "ml_fit_diagnostics.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    run = dict(
        scope="validation_only",
        expected_origins=expected,
        successful_fits=len(fits),
        failures=len(failures),
        winners=winners,
        method_winners=method_winners,
        elapsed_seconds=time.perf_counter() - total_start,
        feature_preparation_seconds=feature_seconds,
        test_evaluated=False,
        grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        data_sha256=settings["processed_sha256"],
        feature_names=FEATURE_NAMES,
        python=sys.version,
        platform=platform.platform(),
        libraries={p: version(p) for p in ["numpy", "scipy", "scikit-learn"]},
    )
    (output / "ml_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2), flush=True)
    for r in summaries:
        if r["fold"] == "all":
            print(r["candidate"], r["horizon"], "RMSE", round(r["RMSE"], 4), flush=True)


if __name__ == "__main__":
    main()
