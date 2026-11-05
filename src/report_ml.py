"""Сводка ML на валидации с обязательными ориентирами naive и SES."""

import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import ROOT, write_csv


def read(name):
    return list(
        csv.DictReader(
            (ROOT / "results/validation" / name).read_text(encoding="utf-8-sig").splitlines()
        )
    )


def main():
    out = ROOT / "results/validation"
    ml = read("ml_metrics.csv")
    base = read("baseline_metrics.csv")
    stats = read("statistical_metrics.csv")
    run = json.loads((out / "ml_run.json").read_text())
    fits = json.loads((out / "ml_fit_diagnostics.json").read_text())
    labels = {
        "ridge": "Ridge",
        "elastic": "Elastic Net",
        "forest": "Random Forest",
        "boost": "Gradient Boosting",
        "svr": "SVR",
    }
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
    for ax, h in zip(axes, ["1", "5"]):
        naive = next(
            r
            for r in base
            if r["candidate"] == "naive" and r["horizon"] == h and r["fold"] == "all"
        )
        ses = next(
            r for r in stats if r["candidate"] == "ses" and r["horizon"] == h and r["fold"] == "all"
        )
        selected = [("Naive", naive), ("SES", ses)] + [
            (
                labels[m],
                next(
                    r
                    for r in ml
                    if r["candidate"] == c and r["horizon"] == h and r["fold"] == "all"
                ),
            )
            for m, c in run["method_winners"][h].items()
        ]
        for label, r in selected:
            table.append(
                dict(
                    horizon=int(h),
                    method=label,
                    candidate=r["candidate"],
                    MAE=float(r["MAE"]),
                    RMSE=float(r["RMSE"]),
                    sMAPE_pct=float(r["sMAPE_pct"]),
                    direction_accuracy_pct=float(r["direction_accuracy_pct"]),
                    RMSE_gain_vs_naive_pct=100 * (1 - float(r["RMSE"]) / float(naive["RMSE"])),
                )
            )
        ax.barh(
            [x[0] for x in selected],
            [float(x[1]["RMSE"]) for x in selected],
            color=["#424c5a", "#71949b"] + ["#347eb1"] * 5,
        )
        ax.axvline(float(naive["RMSE"]), color="#c25142", linestyle="--", linewidth=1.3)
        for i, (_, r) in enumerate(selected):
            ax.text(float(r["RMSE"]) + 0.06, i, f"{float(r['RMSE']):.3f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(float(x[1]["RMSE"]) for x in selected) * 1.16)
        ax.set_title(f"Horizon: {h} trading day(s)")
        ax.set_xlabel("RMSE, USD per barrel (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("ML validation: best setting per method, 104 common origins", fontsize=14)
    figures = ROOT / "results/figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "ml_validation_rmse.png", dpi=170)
    plt.close(fig)
    write_csv(out / "ml_vs_references.csv", table)
    timings = []
    for candidate in sorted(set(x["candidate"] for x in fits)):
        for h in [1, 5]:
            sub = [x for x in fits if x["candidate"] == candidate and x["horizon"] == h]
            if sub:
                timings.append(
                    dict(
                        candidate=candidate,
                        horizon=h,
                        fits=len(sub),
                        total_seconds=sum(x["fit_forecast_seconds"] for x in sub),
                        mean_seconds=sum(x["fit_forecast_seconds"] for x in sub) / len(sub),
                    )
                )
    write_csv(out / "ml_runtime.csv", timings)
    print("Saved ML tables and figure.")


if __name__ == "__main__":
    main()
