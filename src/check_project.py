"""Проверяем ссылки, числа README и отсутствие записи документов из отчётных скриптов."""

import ast
import json
import re

import numpy as np
import pandas as pd
from experiment import ROOT

MODEL_NAMES = {
    "Naive": "naive",
    "SES": "ses",
    "ElasticNet": "elastic_1.0",
    "LSTM": "lstm_8",
    "RNN": "rnn_8",
}


def check_links(root=ROOT):
    missing = []
    for path in [root / "README.md", root / "CHANGELOG.md", *(root / "docs").glob("*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                missing.append(f"{path.name}: {target}")
    if missing:
        raise ValueError("Broken links: " + str(missing))


def check_numbers(root=ROOT):
    primary = pd.read_csv(root / "results/test/primary_metrics.csv")
    for path in [root / "README.md", root / "docs/analysis.md"]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [c.strip().replace("**", "") for c in line.split("|")[1:-1]]
            if len(cells) != 6 or cells[0] not in ["1", "5"] or cells[1] not in MODEL_NAMES:
                continue
            h = int(cells[0])
            cid = MODEL_NAMES[cells[1]]
            expected = primary[(primary.horizon == h) & (primary.candidate == cid)].iloc[0]
            np.testing.assert_allclose(
                [float(c) for c in cells[2:5]],
                [expected.RMSE, expected.MAE, expected.sMAPE_pct],
                rtol=0,
                atol=0.000050001,
            )
            if cid == "naive":
                if cells[5] != "нет сигнала":
                    raise ValueError("Naive must be marked as no direction signal")
            else:
                np.testing.assert_allclose(
                    float(cells[5]), expected.direction_accuracy_pct, rtol=0, atol=0.005001
                )
            rows.append((h, cid))
        if len(rows) != 8 or len(set(rows)) != 8:
            raise ValueError(f"Expected 8 metric rows in {path.name}")
    if len((root / "README.md").read_text(encoding="utf-8").splitlines()) > 170:
        raise ValueError("README should stay compact")
    text = (root / "README.md").read_text(encoding="utf-8")
    raw = pd.read_csv(root / "data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv")
    hourly = json.loads((root / "data/raw/h1_metadata.json").read_text())
    full_review = json.loads((root / "results/audit/full_h1_review.json").read_text())
    facts = {
        "Дневных строк до очистки": len(raw),
        "Исключено плоских записей с нулевым объёмом": int((raw.volume == 0).sum()),
        "Осталось торговых сессий": int((raw.volume > 0).sum()),
        "Часовых записей для полной сверки": hourly["rows"],
        "Торговых дней с отличиями H1/D1": full_review["mismatched_dates"],
    }
    for label, expected in facts.items():
        match = re.search(r"\| " + re.escape(label) + r" \| ([^|]+)\|", text)
        if not match or int(match.group(1).replace("*", "").replace(" ", "")) != expected:
            raise ValueError(f"Wrong data count in README: {label}")


def check_report_writers(root=ROOT):
    for path in (root / "src").glob("*.py"):
        if path.name.startswith("report_") or path.name == "lock_finalists.py":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(
                isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and ("docs/" in n.value or n.value == "docs")
                for n in ast.walk(tree)
            ):
                raise ValueError(f"Report script references docs output: {path.name}")


def main():
    check_links()
    check_numbers()
    check_report_writers()
    print("Project links, README numbers and report destinations: passed")


if __name__ == "__main__":
    main()
