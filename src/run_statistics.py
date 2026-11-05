"""Последовательная валидация статистических моделей; тест не вычисляется."""

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from experiment import ROOT, load_rows, metrics, write_csv
from statistical_models import fit_forecast


def main():
    settings = json.loads((ROOT / "config/experiment.json").read_text(encoding="utf-8"))
    grid_path = ROOT / "config/statistical.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    source = ROOT / "data/processed/brent_active_provisional.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != settings["processed_sha256"]:
        raise ValueError("Dataset changed: review protocol")
    rows = load_rows()[: settings["test_origin_start"] + 1]
    close = [r["close"] for r in rows]
    expected = sum(f["origin_end"] - f["origin_start"] + 1 for f in settings["validation_folds"])
    records, fits, failures = [], [], []
    output = ROOT / "results/validation"
    output.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    for candidate in grid["candidates"]:
        print("Fitting", candidate["id"], flush=True)
        for fold in settings["validation_folds"]:
            for t in range(fold["origin_start"], fold["origin_end"] + 1):
                if t + max(settings["horizons"]) > settings["test_origin_start"]:
                    raise ValueError("Validation target crosses test boundary")
                start = time.perf_counter()
                try:
                    path, diagnostics = fit_forecast(
                        close[: t + 1], max(settings["horizons"]), candidate
                    )
                except Exception as exc:
                    failures.append(
                        {"candidate": candidate["id"], "origin_index": t, "error": repr(exc)}
                    )
                    continue
                elapsed = time.perf_counter() - start
                fits.append(
                    {
                        "candidate": candidate["id"],
                        "origin_index": t,
                        "history_n": t + 1,
                        "fit_and_forecast_seconds": elapsed,
                        **diagnostics,
                    }
                )
                for h in settings["horizons"]:
                    records.append(
                        dict(
                            candidate=candidate["id"],
                            family="statistical",
                            method=candidate["method"],
                            fold=fold["id"],
                            horizon=h,
                            origin_index=t,
                            origin_date=rows[t]["timestamp_utc"][:10],
                            target_index=t + h,
                            target_date=rows[t + h]["timestamp_utc"][:10],
                            current=close[t],
                            actual=close[t + h],
                            predicted=path[h - 1],
                        )
                    )
        (output / "statistical_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
    summaries = []
    for candidate in grid["candidates"]:
        candidate_fits = [f for f in fits if f["candidate"] == candidate["id"]]
        complete = len(candidate_fits) == expected
        # Не сравниваем метрики по удобному подмножеству успешных дат.
        if not complete:
            continue
        for h in settings["horizons"]:
            for fold_id in ["all"] + [f["id"] for f in settings["validation_folds"]]:
                subset = [
                    r
                    for r in records
                    if r["candidate"] == candidate["id"]
                    and r["horizon"] == h
                    and (fold_id == "all" or r["fold"] == fold_id)
                ]
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
    winners = {
        str(h): min(
            [r for r in summaries if r["horizon"] == h and r["fold"] == "all"],
            key=lambda r: (r["RMSE"], r["MAE"], r["candidate"]),
        )["candidate"]
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
                key=lambda r: (r["RMSE"], r["MAE"], r["candidate"]),
            )["candidate"]
            for m in sorted(set(r["method"] for r in summaries))
        }
        for h in settings["horizons"]
    }
    write_csv(output / "statistical_predictions.csv", records)
    write_csv(output / "statistical_metrics.csv", summaries)
    (output / "statistical_fit_diagnostics.json").write_text(
        json.dumps(fits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "scope": "validation_only",
        "expected_origins": expected,
        "successful_fits": len(fits),
        "failures": len(failures),
        "winners": winners,
        "method_winners": method_winners,
        "excluded_incomplete_candidates": sorted(set(f["candidate"] for f in failures)),
        "elapsed_seconds": time.perf_counter() - total_start,
        "test_evaluated": False,
        "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "data_sha256": settings["processed_sha256"],
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {p: version(p) for p in ["numpy", "pandas", "scipy", "statsmodels"]},
    }
    (output / "statistical_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    for r in summaries:
        if r["fold"] == "all":
            print(
                r["candidate"],
                r["horizon"],
                "RMSE",
                round(r["RMSE"], 4),
                "MAE",
                round(r["MAE"], 4),
                flush=True,
            )


if __name__ == "__main__":
    main()
