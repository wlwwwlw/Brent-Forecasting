"""Сверка дневных OHLC с сохранёнными часовыми свечами того же поставщика."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
daily = {
    r["timestamp_utc"][:10]: r
    for r in csv.DictReader((ROOT / "data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv").open())
}
checks = []
for path in sorted((ROOT / "data/validation").glob("h1_*.json")):
    groups = {}
    for row in json.loads(path.read_text()):
        day = datetime.fromtimestamp(row["timestamp"] / 1000, timezone.utc).date().isoformat()
        groups.setdefault(day, []).append(row)
    for day, rows in sorted(groups.items()):
        active = sorted([r for r in rows if r["volume"] > 0], key=lambda r: r["timestamp"])
        if not active:
            match = float(daily[day]["volume"]) == 0
            differences = {}
        else:
            aggregate = dict(
                open=active[0]["open"],
                high=max(r["high"] for r in active),
                low=min(r["low"] for r in active),
                close=active[-1]["close"],
            )
            differences = {k: float(daily[day][k]) - v for k, v in aggregate.items()}
            match = all(abs(x) < 1e-8 for x in differences.values())
        checks.append(
            dict(date=day, active_hours=len(active), match=match, differences=differences)
        )
if not checks or not all(c["match"] for c in checks):
    raise ValueError(f"Cross-frequency verification failed: {checks}")
output = {
    "status": "passed",
    "checks": checks,
    "limitation": "Same supplier, different timeframes; not independent market-price verification. Only sampled dates checked.",
}
(ROOT / "results/cross_frequency_audit.json").write_text(
    json.dumps(output, indent=2), encoding="utf-8"
)
print(
    f"Passed {len(checks)} sampled dates; {sum(c['active_hours'] > 0 for c in checks)} active sessions."
)
