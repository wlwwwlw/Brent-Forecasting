"""Таблицы и график статистического сравнения из сохранённых прогнозов."""

import csv
import json
from collections import Counter
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
    lines = [
        "# Результаты статистических моделей: только валидация",
        "",
        "Сравнены 12 конфигураций пяти подходов. На каждом из 104 origin модель обучается заново на расширяющейся истории; оба горизонта прогнозируются из одной точки. Все конфигурации дали полный набор прогнозов. Финальный тест не рассчитывался.",
        "",
        "| Горизонт | Метод | Конфигурация | MAE | RMSE | sMAPE, % | Направление, % | Выигрыш RMSE к naive, % |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for x in table:
        lines.append(
            f"| {x['horizon']} | {x['method']} | {x['candidate']} | {x['MAE']:.4f} | {x['RMSE']:.4f} | {x['sMAPE_pct']:.3f} | {x['direction_accuracy_pct']:.2f} | {x['RMSE_gain_vs_naive_pct']:.3f} |"
        )
    lines += [
        "",
        "Положительный выигрыш означает меньший RMSE, отрицательный — больший. Показаны лучшие конфигурации каждого подхода, выбранные по этим же валидационным данным; это не несмещённая оценка качества выбранной модели.",
        "",
        "## Устойчивость победителя статистического семейства",
        "",
        "| Горизонт | Блок | RMSE naive | RMSE SES | Выигрыш SES, % |",
        "|---|---|---:|---:|---:|",
    ]
    for h in ["1", "5"]:
        for fold in ["v1", "v2", "v3"]:
            n = next(
                x
                for x in baseline
                if x["candidate"] == "naive" and x["horizon"] == h and x["fold"] == fold
            )
            e = next(
                x
                for x in stat
                if x["candidate"] == "ses" and x["horizon"] == h and x["fold"] == fold
            )
            lines.append(
                f"| {h} | {fold} | {float(n['RMSE']):.4f} | {float(e['RMSE']):.4f} | {100 * (1 - float(e['RMSE']) / float(n['RMSE'])):.3f} |"
            )
    alpha = [x["smoothing_parameters"]["smoothing_level"] for x in fits if x["candidate"] == "ses"]
    lines += [
        "",
        "## Что можно заключить",
        "",
        "- На h=1 naive лучше всех рассмотренных статистических конфигураций по суммарному RMSE.",
        "- На h=5 SES имеет минимальный RMSE в статистическом семействе и улучшает naive примерно на 0.24%. При этом MAE у SES немного выше: преимущество зависит от критерия.",
        "- На h=5 SES выигрывает только на одном из трёх блоков. Поэтому результат не доказывает устойчивое или статистически значимое превосходство.",
        "- Причинный вывод о том, почему конкретная модель проиграла, по одной таблице невозможен. Возможные объяснения — слабая предсказуемость изменений, ошибки оценки и запаздывание; их следует называть гипотезами.",
        f"- Параметр alpha у SES находится в диапазоне {min(alpha):.4f}–{max(alpha):.4f}. Метод придаёт большой вес последней цене, но не совпадает с naive на всех origin.",
        "- Нулевая точность направления naive в этой таблице связана с прогнозом нулевого изменения при реально ненулевых изменениях. Его покрытие направленного сигнала равно нулю. У SES микроскопические отличия от последней цены могут формально создавать знак; направленная точность сама по себе не подтверждает экономически полезный сигнал.",
        "",
        "## Вычисления и надёжность",
        "",
        f"Успешных обучений: {len(fits)}. Повторов оптимизации: {sum(x['attempts'] > 1 for x in fits)}. Ошибок после предусмотренного повтора: {run['failures']}. Время полного поиска: {run['elapsed_seconds']:.2f} секунд на данной машине, без установки библиотек. Оно не является гарантией времени на другом оборудовании.",
        "",
        "Предупреждения сохранены, а не скрыты:",
        "",
    ]
    for message, count in Counter(w for x in fits for w in x["warnings"]).items():
        lines.append(f"- {count}: {message}")
    lines += [
        "",
        "Предупреждения об исходных AR/MA-параметрах относятся к инициализации оптимизации. Предупреждения о несходимости относятся к первой попытке; итоговый статус повторной попытки проверен.",
        "",
        "График: `../results/figures/statistical_validation_rmse.png`. Полные метрики, прогнозы и диагностика: `../results/validation/`.",
    ]
    (ROOT / "docs/statistical_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved statistical comparison, runtime table, report and figure.")


if __name__ == "__main__":
    main()
