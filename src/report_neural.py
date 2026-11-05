"""Нейросетевые результаты: средние метрики seed, без выбора лучшего seed."""

import csv, json
from statistics import mean
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import ROOT, write_csv
from checkpoint import effective_entries


def read(name):
    return list(
        csv.DictReader(
            (ROOT / "results/validation" / name).read_text(encoding="utf-8-sig").splitlines()
        )
    )


def main():
    out = ROOT / "results/validation"
    rows = read("neural_metrics.csv")
    base = read("baseline_metrics.csv")
    stats = read("statistical_metrics.csv")
    ml = read("ml_metrics.csv")
    run = json.loads((out / "neural_run.json").read_text())
    ml_run = json.loads((out / "ml_run.json").read_text())
    entries = list(
        effective_entries(
            [json.loads(l) for l in (out / "neural_checkpoint.jsonl").read_text().splitlines() if l]
        ).values()
    )
    good = [r for r in entries if r["status"] == "ok"]
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for ax, h in zip(axes, ["1", "5"]):
        n = next(
            r
            for r in base
            if r["candidate"] == "naive" and r["horizon"] == h and r["fold"] == "all"
        )
        sub = [r for r in rows if r["horizon"] == h and r["fold"] == "all"]
        labels = ["Naive"] + [r["architecture"].upper() for r in sub]
        values = [float(n["RMSE"])] + [float(r["RMSE_mean"]) for r in sub]
        sd = [0] + [float(r["RMSE_seed_sd"]) for r in sub]
        ax.barh(
            labels,
            values,
            color=["#424c5a"] + ["#7a5a9b"] * len(sub),
            xerr=sd,
            capsize=3,
            error_kw={"ecolor": "#292929"},
        )
        ax.axvline(float(n["RMSE"]), linestyle="--", color="#c25142")
        ax.set_xlim(0, max(v + s for v, s in zip(values, sd)) * 1.18)
        for i, (v, s) in enumerate(zip(values, sd)):
            ax.text(v + s + 0.08, i, f"{v:.3f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Horizon: {h} trading day(s)")
        ax.set_xlabel("RMSE, USD/barrel; error bars: SD across 3 seeds")
        ax.spines[["top", "right"]].set_visible(False)
        for r in sub:
            table.append(
                dict(
                    horizon=int(h),
                    architecture=r["architecture"],
                    candidate=r["candidate"],
                    RMSE_mean=float(r["RMSE_mean"]),
                    RMSE_seed_sd=float(r["RMSE_seed_sd"]),
                    MAE_mean=float(r["MAE_mean"]),
                    sMAPE_pct_mean=float(r["sMAPE_pct_mean"]),
                    direction_accuracy_pct_mean=float(r["direction_accuracy_pct_mean"]),
                    gain_vs_naive_pct=100 * (1 - float(r["RMSE_mean"]) / float(n["RMSE"])),
                )
            )
    fig.suptitle("Neural validation: 104 origins, seeds 7 / 42 / 2026", fontsize=14)
    (ROOT / "results/figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results/figures/neural_validation_rmse.png", dpi=170)
    plt.close(fig)
    write_csv(out / "neural_vs_naive.csv", table)
    timings = []
    for c in sorted(set(r["candidate"] for r in good)):
        for h in [1, 5]:
            for seed in run["seeds"]:
                sub = [
                    r
                    for r in good
                    if r["candidate"] == c and r["horizon"] == h and r["seed"] == seed
                ]
                timings.append(
                    dict(
                        candidate=c,
                        horizon=h,
                        seed=seed,
                        fits=len(sub),
                        total_fit_seconds=sum(r["fit_forecast_seconds"] for r in sub),
                        mean_fit_seconds=mean(r["fit_forecast_seconds"] for r in sub),
                        parameters=sub[0]["diagnostics"]["parameter_count"],
                        training_loss_increased=sum(
                            r["diagnostics"]["final_train_mse_standardized"]
                            > r["diagnostics"]["initial_train_mse_standardized"]
                            for r in sub
                        ),
                    )
                )
    write_csv(out / "neural_runtime.csv", timings)
    comparison = []
    for h, c in run["winners"].items():
        for label, data, candidate in [
            ("Naive", base, "naive"),
            ("Statistical", stats, "ses"),
            ("ML", ml, ml_run["winners"][h]),
        ]:
            r = next(
                r
                for r in data
                if r["horizon"] == h and r["candidate"] == candidate and r["fold"] == "all"
            )
            comparison.append(
                dict(
                    horizon=int(h),
                    family=label,
                    candidate=candidate,
                    RMSE=float(r["RMSE"]),
                    MAE=float(r["MAE"]),
                    aggregation="one deterministic/fixed-seed run",
                )
            )
        r = next(
            r for r in rows if r["horizon"] == h and r["candidate"] == c and r["fold"] == "all"
        )
        comparison.append(
            dict(
                horizon=int(h),
                family="Neural",
                candidate=c,
                RMSE=float(r["RMSE_mean"]),
                MAE=float(r["MAE_mean"]),
                aggregation="mean of three seed-specific metrics",
            )
        )
    write_csv(out / "family_validation_summary.csv", comparison)
    print("Saved neural and family validation summaries, diagnostics and figure.")
    for r in comparison:
        print(r)


if __name__ == "__main__":
    main()
