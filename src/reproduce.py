"""Проверяем копию проекта; опубликованные данные и результаты не перезаписываем."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from experiment import ROOT


def compare(original, rebuilt, prediction=False):
    left, right = pd.read_csv(original), pd.read_csv(rebuilt)
    keys = ["candidate", "horizon"]
    if prediction:
        keys += ["origin_index"]
    for key in ["seed", "fold", "period", "block_length"]:
        if key in left:
            keys.append(key)
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[keys], right[keys])
    columns = (
        ["actual", "current", "predicted"]
        if prediction
        else [
            c
            for c in [
                "MAE",
                "RMSE",
                "sMAPE_pct",
                "direction_accuracy_pct",
                "direction_coverage_pct",
                "RMSE_difference",
                "lower_95",
                "upper_95",
            ]
            if c in left
        ]
    )
    tolerance = 1e-6 if prediction else 1e-10
    np.testing.assert_allclose(left[columns], right[columns], rtol=0, atol=tolerance)
    return dict(
        rows=len(left),
        max_difference=float(np.max(abs(left[columns].to_numpy() - right[columns].to_numpy()))),
        tolerance=tolerance,
    )


def run_script(root, name):
    subprocess.run([sys.executable, str(root / "src" / name)], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain", action="store_true", help="Повторить обучение в отдельной копии"
    )
    parser.add_argument("--output", type=Path, help="Сохранить итог проверки в указанном JSON")
    args = parser.parse_args()
    report = dict(
        python=sys.version,
        libraries={
            p: version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "torch"]
        },
    )
    with tempfile.TemporaryDirectory(prefix="brent_check_") as directory:
        dest = Path(directory) / "project"
        shutil.copytree(
            ROOT,
            dest,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "node_modules",
                "__pycache__",
                ".ruff_cache",
                ".pytest_cache",
                "work",
                "recheck",
            ),
        )
        for script in ["audit_data.py", "prepare_data.py", "verify_hourly.py", "final_audit.py"]:
            run_script(dest, script)
        report["hourly_review"] = json.loads(
            (dest / "results/audit/full_h1_review.json").read_text()
        )
        if args.retrain:
            for name in ["neural_checkpoint.jsonl", "neural_checkpoint_meta.json"]:
                (dest / "results/validation" / name).unlink()
            for script in [
                "run_final.py",
                "run_baselines.py",
                "run_statistics.py",
                "run_ml.py",
                "run_neural.py",
            ]:
                run_script(dest, script)
            names = ["results/test/predictions.csv"] + [
                f"results/validation/{f}_predictions.csv"
                for f in ["baseline", "statistical", "ml", "neural"]
            ]
            report["retraining"] = {name: compare(ROOT / name, dest / name, True) for name in names}
        for script in [
            "report_statistics.py",
            "report_ml.py",
            "report_neural.py",
            "report_final.py",
        ]:
            run_script(dest, script)
        names = ["metrics.csv", "primary_metrics.csv", "paired_bootstrap.csv"]
        report["recalculated_tables"] = {
            name: compare(ROOT / "results/test" / name, dest / "results/test" / name)
            for name in names
        }
        run_script(dest, "check_project.py")
        run = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=dest,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        report["tests"] = dict(exit_code=run.returncode, output=run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
    report["status"] = "passed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
