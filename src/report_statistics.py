"""Таблицы и график статистического сравнения из сохранённых прогнозов."""

import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import ROOT, write_csv


def main():
    out = ROOT / "results/validation"
    baseline = list(
        csv.DictReader((out / "baseline_metrics.csv").read_text(encoding="utf-8-sig").splitlines())
    )
    stat = list(
        csv.DictReader(
            (out / "statistical_metrics.csv").read_text(encoding="utf-8-sig").splitlines()
        )
    )
    run = json.loads((out / "statistical_run.json").read_text())
    fits = json.loads((out / "statistical_fit_diagnostics.json").read_text())
    labels = {
        "ar": "AR",
        "arima": "ARIMA",
        "ses": "SES",
        "holt": "Holt",
        "holt_damped": "Damped Holt",
    }
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, h in zip(axes, ["1", "5"]):
        ref = next(
            x
            for x in baseline
            if x["candidate"] == "naive" and x["horizon"] == h and x["fold"] == "all"
        )
        selected = [("Naive", ref)]
        for method, candidate in run["method_winners"][h].items():
            row = next(
                x
                for x in stat
                if x["candidate"] == candidate and x["horizon"] == h and x["fold"] == "all"
            )
            selected.append((labels[method], row))
        for label, row in selected:
            table.append(
                dict(
                    horizon=int(h),
                    method=label,
                    candidate=row["candidate"],
                    MAE=float(row["MAE"]),
                    RMSE=float(row["RMSE"]),
                    sMAPE_pct=float(row["sMAPE_pct"]),
                    direction_accuracy_pct=float(row["direction_accuracy_pct"]),
                    direction_coverage_pct=float(row["direction_coverage_pct"]),
                    RMSE_gain_vs_naive_pct=100 * (1 - float(row["RMSE"]) / float(ref["RMSE"])),
                )
            )
        ax.barh(
            [x[0] for x in selected],
            [float(x[1]["RMSE"]) for x in selected],
            color=["#424c5a"] + ["#2e79ae"] * 5,
        )
        ax.axvline(
            float(ref["RMSE"]),
            color="#c25142",
            linestyle="--",
            linewidth=1.3,
            label="Naive reference",
        )
        for i, (_, row) in enumerate(selected):
            ax.text(
                float(row["RMSE"]) + 0.07, i, f"{float(row['RMSE']):.3f}", va="center", fontsize=10
            )
        ax.invert_yaxis()
        ax.set_xlim(0, max(float(x[1]["RMSE"]) for x in selected) * 1.18)
        ax.set_title(f"Horizon: {h} trading day(s)")
        ax.set_xlabel("RMSE, USD per barrel (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Validation only: 104 common forecast origins", fontsize=15)
    figures = ROOT / "results/figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "statistical_validation_rmse.png", dpi=170)
    plt.close(fig)
    write_csv(out / "statistical_vs_naive.csv", table)
    runtime = []
    for candidate in sorted(set(x["candidate"] for x in fits)):
        subset = [x for x in fits if x["candidate"] == candidate]
        runtime.append(
            dict(
                candidate=candidate,
                fits=len(subset),
                total_fit_forecast_seconds=sum(x["fit_and_forecast_seconds"] for x in subset),
                mean_fit_forecast_seconds=sum(x["fit_and_forecast_seconds"] for x in subset)
                / len(subset),
                retries=sum(x["attempts"] > 1 for x in subset),
                warning_fits=sum(bool(x["warnings"]) for x in subset),
            )
        )
    write_csv(out / "statistical_runtime.csv", runtime)
    print("Saved statistical tables and figure.")


if __name__ == "__main__":
    main()
