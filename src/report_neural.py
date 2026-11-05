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
    seeds = read("neural_seed_metrics.csv")
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
    lines = [
        "# Нейросети: результаты валидации",
        "",
        "Пять архитектур, три фиксированных seed, два горизонта. Среднее и SD ниже рассчитаны по метрикам трёх отдельных запусков, а не по ансамблевому прогнозу. SD seed не является доверительным интервалом по будущим датам. Финальный тест не использован.",
        "",
        "| Горизонт | Архитектура | RMSE среднее | SD seed | MAE среднее | sMAPE среднее, % | Направление среднее, % | Выигрыш RMSE к naive, % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['horizon']} | {r['architecture'].upper()} | {r['RMSE_mean']:.4f} | {r['RMSE_seed_sd']:.4f} | {r['MAE_mean']:.4f} | {r['sMAPE_pct_mean']:.3f} | {r['direction_accuracy_pct_mean']:.2f} | {r['gain_vs_naive_pct']:.3f} |"
        )
    lines += [
        "",
        "## Выбранная архитектура и её отдельные seed",
        "",
        "| Горизонт | Архитектура | Seed | RMSE | MAE |",
        "|---|---|---|---:|---:|",
    ]
    for h, c in run["winners"].items():
        for r in seeds:
            if r["horizon"] == h and r["candidate"] == c and r["fold"] == "all":
                lines.append(
                    f"| {h} | {c} | {r['seed']} | {float(r['RMSE']):.4f} | {float(r['MAE']):.4f} |"
                )
    lines += [
        "",
        "## Устойчивость по времени",
        "",
        "| Горизонт | Блок | RMSE naive | RMSE победителя, среднее seed | Выигрыш, % |",
        "|---|---|---:|---:|---:|",
    ]
    for h, c in run["winners"].items():
        for fold in ["v1", "v2", "v3"]:
            n = next(
                r
                for r in base
                if r["candidate"] == "naive" and r["horizon"] == h and r["fold"] == fold
            )
            r = next(
                r for r in rows if r["candidate"] == c and r["horizon"] == h and r["fold"] == fold
            )
            lines.append(
                f"| {h} | {fold} | {float(n['RMSE']):.4f} | {float(r['RMSE_mean']):.4f} | {100 * (1 - float(r['RMSE_mean']) / float(n['RMSE'])):.3f} |"
            )
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
    lines += [
        "",
        "## Ограничения выводов",
        "",
        "- Отбор архитектуры выполняется по среднему RMSE трёх seed. Самый удачный seed не выбирается для презентации как представитель семейства.",
        "- Валидационные оценки использованы для выбора, поэтому ещё не доказывают качество на независимом периоде.",
        "- Фиксированные 60 эпох не гарантируют одинаковой сходимости или глобального минимума разных архитектур. Одна небольшая конфигурация не представляет весь класс нейросетей.",
        "- Снижение training loss не означает улучшения будущего прогноза. Сравнивать нужно сохранённые внеобучающие ошибки.",
        "- У ML и нейросетей различается представление входа, у статистических методов также отличается способ многопрогнозного расчёта. Сравниваются полные процедуры, а не изолированные архитектуры.",
        "",
        "## Вычисления",
        "",
        f"Успешных обучений: {run['successful_fits']}. Сбоев: {run['failed_fits']}. Сумма времени fit+forecast: {run['total_fit_seconds']:.2f} секунд. Затраты одного seed и всего исследования приведены отдельно в neural_runtime.csv.",
        "",
    ]
    for c in sorted(set(r["candidate"] for r in good)):
        sub = [r for r in good if r["candidate"] == c]
        increased = sum(
            r["diagnostics"]["final_train_mse_standardized"]
            > r["diagnostics"]["initial_train_mse_standardized"]
            for r in sub
        )
        lines.append(
            f"- {c}: {sub[0]['diagnostics']['parameter_count']} параметров; рост конечного training loss относительно начального в {increased}/{len(sub)} обучениях."
        )
    (ROOT / "docs/neural_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved neural and family validation summaries, diagnostics and figure.")
    for r in comparison:
        print(r)


if __name__ == "__main__":
    main()
