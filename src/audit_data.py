"""Воспроизводимый первичный аудит. Запуск: python src/audit_data.py."""

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv"
FIELDS = ["timestamp_utc", "open", "high", "low", "close", "volume"]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with RAW.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected CSV schema")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty dataset")
    dates, excluded, active = [], [], []
    for row in rows:
        date = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
        dates.append(date)
        for field in FIELDS[1:]:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f"Nonfinite {field}: {date}")
        o, h, l, c, v = (row[k] for k in FIELDS[1:])
        if not (0 < l <= min(o, c) <= max(o, c) <= h and v >= 0):
            raise ValueError(f"Invalid OHLC/volume: {date}")
        if v == 0:
            # Нулевой объём сам по себе недостаточен: проверяем и отсутствие движения.
            if not o == h == l == c:
                raise ValueError(f"Zero-volume nonflat candle needs review: {date}")
            excluded.append({**row, "reason": "zero_volume_and_flat_ohlc"})
        else:
            active.append(row)
    if dates != sorted(set(dates)):
        raise ValueError("Unsorted or duplicate timestamps")
    expected = [dates[0] + timedelta(days=i) for i in range(365)]
    if dates != expected or dates[0].isoformat() != "2025-10-28T00:00:00+00:00":
        raise ValueError("Unexpected dates or missing calendar rows")
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    metadata = json.loads((RAW.parent / "metadata.json").read_text(encoding="utf-8"))
    if digest != metadata["sha256"]:
        raise ValueError("Raw file checksum changed")
    report = {
        "status": "structural_checks_passed_external_calendar_review_pending",
        "raw_rows": len(rows),
        "active_rows": len(active),
        "excluded_flat_zero_volume": len(excluded),
        "excluded_weekdays": dict(
            Counter(
                datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00")).strftime("%A")
                for r in excluded
            )
        ),
        "excluded_monday_to_friday": [
            r["timestamp_utc"]
            for r in excluded
            if datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00")).weekday() < 5
        ],
        "duplicates": 0,
        "missing_calendar_dates": 0,
        "invalid_ohlc": 0,
        "nonfinite_values": 0,
        "raw_sha256": digest,
        "limitations": [
            "Historical Brent-specific calendar and adjustment amounts not independently verified.",
            "No interpolation, clipping, smoothing or outlier removal.",
            "Volume is provider-specific; not verified exchange volume.",
            "Active observations are provisional trading sessions.",
        ],
    }
    write_csv(ROOT / "data/processed/brent_active_provisional.csv", active, FIELDS)
    write_csv(ROOT / "results/excluded_rows.csv", excluded, FIELDS + ["reason"])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/data_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
