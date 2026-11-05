"""Только валидация простых методов. Финальный тест здесь не вычисляется."""

import hashlib
import json
import time
from experiment import ROOT, load_rows, metrics, forecast_simple, write_csv


def main():
    config = json.loads((ROOT / "config/experiment.json").read_text(encoding="utf-8"))
    source = ROOT / "data/processed/brent_active_provisional.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["processed_sha256"]:
        raise ValueError("Dataset changed: review and version protocol before proceeding")
    rows = load_rows()
    # Физически отсекаем тестовые цены от входа этапа выбора моделей.
    development = rows[: config["test_origin_start"] + 1]
    close = [r["close"] for r in development]
    records = []
    for candidate in config["baseline_candidates"]:
        for h in config["horizons"]:
            for fold in config["validation_folds"]:
                for t in range(fold["origin_start"], fold["origin_end"] + 1):
                    if t + h > config["test_origin_start"]:
                        raise ValueError("Validation target reaches test")
                    start = time.perf_counter()
                    pred = forecast_simple(
                        close[: t + 1], h, candidate["model"], candidate.get("window")
                    )
                    elapsed = time.perf_counter() - start
                    records.append(
                        dict(
                            candidate=candidate["id"],
                            family="simple",
                            fold=fold["id"],
                            horizon=h,
                            origin_index=t,
                            origin_date=development[t]["timestamp_utc"][:10],
                            target_index=t + h,
                            target_date=development[t + h]["timestamp_utc"][:10],
                            current=close[t],
                            actual=close[t + h],
                            predicted=pred,
                            forecast_seconds=elapsed,
                        )
                    )
    summary = []
    for candidate in config["baseline_candidates"]:
        for h in config["horizons"]:
            for fold_id in ["all"] + [f["id"] for f in config["validation_folds"]]:
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
                summary.append(
                    dict(
                        candidate=candidate["id"],
                        horizon=h,
                        fold=fold_id,
                        **score,
                        total_forecast_seconds=sum(r["forecast_seconds"] for r in subset),
                    )
                )
    winners = {
        str(h): min(
            [r for r in summary if r["horizon"] == h and r["fold"] == "all"],
            key=lambda r: (r["RMSE"], r["MAE"], r["candidate"]),
        )["candidate"]
        for h in config["horizons"]
    }
    output = ROOT / "results/validation"
    write_csv(output / "baseline_predictions.csv", records)
    write_csv(output / "baseline_metrics.csv", summary)
    (output / "simple_winners.json").write_text(
        json.dumps(
            {
                "scope": "validation_only",
                "primary_metric": "RMSE",
                "winners": winners,
                "naive_always_retained": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "validation_origins_per_horizon": sum(
                    f["origin_end"] - f["origin_start"] + 1 for f in config["validation_folds"]
                ),
                "winners": winners,
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    for r in summary:
        if r["fold"] == "all":
            print(
                r["candidate"],
                r["horizon"],
                "RMSE",
                round(r["RMSE"], 4),
                "MAE",
                round(r["MAE"], 4),
                "sMAPE",
                round(r["sMAPE_pct"], 3),
                "DA",
                round(r["direction_accuracy_pct"], 2),
            )


if __name__ == "__main__":
    main()
