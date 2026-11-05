"""Сводка ML на валидации с обязательными ориентирами naive и SES."""

import csv
import json
from collections import Counter
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
    lines = [
        "# ML: результаты только валидации",
        "",
        "15 конфигураций пяти методов. Лучшие настройки выбраны на этих же 104 origin; это не независимое подтверждение их преимущества. Финальный тест не рассчитывался.",
        "",
        "| Горизонт | Метод | Конфигурация | MAE | RMSE | sMAPE, % | Направление, % | Выигрыш RMSE к naive, % |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['horizon']} | {r['method']} | {r['candidate']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['sMAPE_pct']:.3f} | {r['direction_accuracy_pct']:.2f} | {r['RMSE_gain_vs_naive_pct']:.3f} |"
        )
    lines += [
        "",
        "## Победители ML по временным блокам",
        "",
        "| Горизонт | Кандидат | Блок | RMSE naive | RMSE ML | Выигрыш, % |",
        "|---|---|---|---:|---:|---:|",
    ]
    for h, c in run["winners"].items():
        for fold in ["v1", "v2", "v3"]:
            n = next(
                r
                for r in base
                if r["candidate"] == "naive" and r["horizon"] == h and r["fold"] == fold
            )
            m = next(
                r for r in ml if r["candidate"] == c and r["horizon"] == h and r["fold"] == fold
            )
            lines.append(
                f"| {h} | {c} | {fold} | {float(n['RMSE']):.4f} | {float(m['RMSE']):.4f} | {100 * (1 - float(m['RMSE']) / float(n['RMSE'])):.3f} |"
            )
    lines += [
        "",
        "## Как трактовать",
        "",
        "- Положительный выигрыш означает меньший RMSE. Победитель внутри ML не обязан превосходить naive.",
        "- Для утверждения об устойчивом улучшении нужно учитывать все блоки, другие метрики и независимый финальный тест. Пока результат является отборочным.",
        "- Одного seed недостаточно для вывода об устойчивости случайного леса. Дополнительные seeds выбранных случайных моделей следует проверять без повторного выбора удачного seed.",
        "- Модели обучаются на изменении цены, а ошибки измеряются после восстановления уровня. Прямой и рекурсивный способы прогноза различаются, поэтому нельзя приписывать весь эффект только названию алгоритма.",
        "- Нулевая направленная точность naive означает отсутствие направленного сигнала при ненулевых фактических изменениях, а не случайное угадывание с нулевой вероятностью успеха.",
        "",
        "## Выполнение",
        "",
        f"Обучений: {run['successful_fits']}; ошибок: {run['failures']}. Полный запуск: {run['elapsed_seconds']:.2f} сек.; общая подготовка признаков: {run['feature_preparation_seconds']:.2f} сек. Время зависит от компьютера и нагрузки. Затраты двух горизонтов у ML суммируются, поскольку модели обучаются отдельно.",
        "",
        f"Размеры обучающих выборок: от {min(x['training_n'] for x in fits)} до {max(x['training_n'] for x in fits)}. Каждый обучающий target имеет индекс не больше origin.",
        "",
    ]
    warnings = Counter(w for x in fits for w in x["warnings"])
    lines.append("Предупреждения: " + (str(dict(warnings)) if warnings else "нет."))
    (ROOT / "docs/ml_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved ML report, tables and figure.")


if __name__ == "__main__":
    main()
