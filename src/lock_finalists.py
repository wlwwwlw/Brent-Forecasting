"""Summarize validation stability and lock the final evaluation before running it."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from experiment import ROOT, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "results/validation"
    families = {
        "simple": "simple_winners.json",
        "statistical": "statistical_run.json",
        "ml": "ml_run.json",
        "neural": "neural_run.json",
    }
    winners = {f: json.loads((out / name).read_text())["winners"] for f, name in families.items()}
    stability = []
    for family, filename in [
        ("simple", "baseline_metrics.csv"),
        ("statistical", "statistical_metrics.csv"),
        ("ml", "ml_metrics.csv"),
        ("neural", "neural_metrics.csv"),
    ]:
        with (out / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        for h in ["1", "5"]:
            for row in rows:
                if row["horizon"] != h or row["candidate"] != winners[family][h]:
                    continue
                stability.append(
                    dict(
                        family=family,
                        candidate=row["candidate"],
                        horizon=int(h),
                        fold=row["fold"],
                        RMSE=float(row.get("RMSE", row.get("RMSE_mean"))),
                        MAE=float(row.get("MAE", row.get("MAE_mean"))),
                    )
                )
    write_csv(out / "finalist_stability.csv", stability)
    lines = [
        "# Устойчивость выбранных моделей на валидации",
        "",
        "Модели выбраны по всей валидации; блоки ниже — диагностика, а не независимый тест.",
        "",
        "| Горизонт | Модель | RMSE v1 | RMSE v2 | RMSE v3 | Побед над naive |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for h in [1, 5]:
        for family in families:
            selected = [
                r
                for r in stability
                if r["horizon"] == h and r["family"] == family and r["fold"] != "all"
            ]
            selected.sort(key=lambda r: r["fold"])
            base = {
                r["fold"]: r["RMSE"]
                for r in stability
                if r["horizon"] == h and r["family"] == "simple"
            }
            wins = sum(r["RMSE"] < base[r["fold"]] for r in selected)
            lines.append(
                f"| {h} | {winners[family][str(h)]} | "
                + " | ".join(f"{r['RMSE']:.4f}" for r in selected)
                + f" | {wins}/3 |"
            )
    lines += [
        "",
        "Для нейросетей приведены средние метрики трёх seed, а не прогноз ансамбля.",
        "Разброс по seed сохранён в neural_metrics.csv. Три seed не дают надёжной оценки всей случайности обучения.",
        "Проверка блоков не меняет уже установленное правило выбора по общему RMSE.",
        "Отдельный повторный перебор seed леса/бустинга не выполнен; победитель ML ElasticNet детерминирован при фиксированных данных.",
        "Устойчивость к другим годам, поставщикам, представлениям признаков и сеткам параметров не установлена.",
    ]
    (ROOT / "docs/validation_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock_path = ROOT / "config/final_lock.json"
    if lock_path.exists():
        print("Existing final lock retained; stability report refreshed.")
        return
    if (ROOT / "results/test/predictions.csv").exists():
        raise RuntimeError("Cannot claim pre-test lock after test predictions exist")
    paths = [
        ROOT / "config" / name
        for name in ["experiment.json", "statistical.json", "ml.json", "neural.json"]
    ]
    paths += [
        ROOT / "src" / name
        for name in [
            "experiment.py",
            "statistical_models.py",
            "ml_models.py",
            "neural_models.py",
            "run_final.py",
            "report_final.py",
        ]
    ]
    paths += [ROOT / "data/processed/brent_active_provisional.csv"]
    paths += [out / name for name in families.values()]
    lock = dict(
        locked_at_utc=datetime.now(timezone.utc).isoformat(),
        status="locked_before_final_evaluation",
        winners=winners,
        primary_metric="RMSE",
        neural_reference_seed=42,
        neural_seeds=[7, 42, 2026],
        neural_reporting="primary fixed seed 42; all seed metrics and their mean separately; no ensemble",
        refit="expanding history each origin; only labels available at that origin",
        test_blocks=[[207, 229], [230, 252]],
        bootstrap=dict(
            method="paired circular block percentile bootstrap of saved errors",
            block_lengths=[5, 10],
            primary_block_length=5,
            replicates=10000,
            seed=20261104,
            confidence=0.95,
            simultaneous=False,
            comparisons="each non-naive finalist versus naive, fixed seed 42",
        ),
        inference="exploratory pointwise intervals; short dependent sample, possible nonstationarity and multiple comparisons limit claims; no tuning after test",
        hashes={str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths},
    )
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
