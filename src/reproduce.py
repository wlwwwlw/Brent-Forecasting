"""Check release artifacts or rebuild predictions in a separate temporary copy."""

import argparse
import hashlib
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


def compare(original, rebuilt):
    left = pd.read_csv(original)
    right = pd.read_csv(rebuilt)
    keys = ["candidate", "horizon", "origin_index"] + (["seed"] if "seed" in left else [])
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[keys], right[keys])
    for column in ["current", "actual"]:
        np.testing.assert_allclose(left[column], right[column], rtol=0, atol=1e-12)
    np.testing.assert_allclose(left.predicted, right.predicted, rtol=0, atol=1e-6)
    return dict(
        rows=len(left),
        max_absolute_prediction_difference=float(abs(left.predicted - right.predicted).max()),
        tolerance_USD=1e-6,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Rebuild all models in an isolated copy; takes several minutes",
    )
    args = parser.parse_args()
    report = {
        "python": sys.version,
        "libraries": {
            p: version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "torch"]
        },
    }
    if args.retrain:
        with tempfile.TemporaryDirectory(prefix="brent_reproduction_") as temporary:
            dest = Path(temporary)
            for name in ["src", "config", "data", "results", "provenance"]:
                shutil.copytree(
                    ROOT / name,
                    dest / name,
                    ignore=shutil.ignore_patterns("__pycache__", "recheck"),
                )
            checkpoint = dest / "results/validation/neural_checkpoint.jsonl"
            checkpoint.unlink()
            (dest / "results/validation/neural_checkpoint_meta.json").unlink()
            for script in [
                "run_final.py",
                "run_baselines.py",
                "run_statistics.py",
                "run_ml.py",
                "run_neural.py",
            ]:
                subprocess.run([sys.executable, str(dest / "src" / script)], cwd=dest, check=True)
            pairs = ["results/test/predictions.csv"] + [
                f"results/validation/{name}_predictions.csv"
                for name in ["baseline", "statistical", "ml", "neural"]
            ]
            report["independent_retraining"] = {
                name: compare(ROOT / name, dest / name) for name in pairs
            }
    for script in ["audit_data.py", "verify_hourly.py", "final_audit.py"]:
        subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)
    run = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report["unit_tests"] = dict(exit_code=run.returncode, output=run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(report["unit_tests"]["output"])
    report["status"] = "passed"
    report["raw_sha256"] = hashlib.sha256(
        (ROOT / "data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv").read_bytes()
    ).hexdigest()
    target = ROOT / "results/audit" / ("retraining.json" if args.retrain else "release_checks.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
