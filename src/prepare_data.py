"""Сверка всего D1-ряда с завершёнными H1-свечами. Исходные цены не меняет."""

import hashlib
import json

import numpy as np
import pandas as pd

from experiment import ROOT

OHLC = ["open", "high", "low", "close"]
H1_FILE = "data/raw/brent_h1_bid_2025-10-28_2026-10-28.json"


def hourly_frame(records):
    frame = pd.DataFrame(records)
    required = ["timestamp", *OHLC, "volume"]
    if not len(frame) or not set(required) <= set(frame.columns):
        raise ValueError("Empty hourly archive or invalid schema")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite hourly value")
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted hourly timestamps")
    if (frame.timestamp % 3600000 != 0).any():
        raise ValueError("Hourly timestamp must be the start of a UTC hour")
    valid = (
        (frame.low > 0)
        & (frame.low <= frame[["open", "close"]].min(axis=1))
        & (frame.high >= frame[["open", "close"]].max(axis=1))
        & (frame.volume >= 0)
    )
    if not valid.all():
        raise ValueError("Invalid hourly OHLC or volume")
    flat = frame[OHLC].nunique(axis=1) == 1
    if ((frame.volume == 0) & ~flat).any():
        raise ValueError("Zero-volume moving candle needs review")
    frame["start"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    frame["end"] = frame.start + pd.Timedelta(hours=1)
    frame["date"] = frame.start.dt.strftime("%Y-%m-%d")
    return frame


def rebuild_daily(frame, cutoff):
    active = frame[(frame.volume > 0) & (frame.end <= pd.Timestamp(cutoff))]
    return active.groupby("date").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )


def review(root=ROOT, write=True):
    settings = json.loads((root / "config/data_review.json").read_text())
    cutoff = pd.Timestamp(settings["cutoff_utc"])
    if pd.Timestamp(settings["assignment_time"]) != cutoff:
        raise ValueError("Assignment and UTC cutoff disagree")
    source = root / H1_FILE
    metadata = json.loads((root / "data/raw/h1_metadata.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Hourly archive checksum changed")
    frame = hourly_frame(json.loads(source.read_text()))
    if len(frame) != metadata["rows"]:
        raise ValueError("Hourly archive count changed")
    expected = pd.date_range(
        settings["from_utc"], settings["to_utc_exclusive"], inclusive="left", freq="h"
    )
    if not frame.start.isin(expected).all():
        raise ValueError("Hourly data outside requested range")
    missing = expected.difference(frame.start)
    daily = pd.read_csv(root / "data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv")
    daily["date"] = daily.timestamp_utc.str[:10]
    rebuilt = rebuild_daily(frame, cutoff)
    active = frame[frame.volume > 0]
    future = active[active.end > cutoff]
    if len(future):
        raise ValueError("Active prices formed after assignment cutoff")
    sessions = []
    for date, rows in active.groupby("date"):
        start, end = rows.start.min(), rows.end.max()
        interior = pd.date_range(start, end, inclusive="left", freq="h").difference(rows.start)
        rule = [
            r for r in settings["nominal_session_changes_observed_in_archive"] if r["from"] <= date
        ][-1]
        shortened = end.hour < rule["end_hour_utc"]
        if len(interior) or start.hour != rule["start_hour_utc"] or end.hour > rule["end_hour_utc"]:
            raise ValueError(f"Unexpected session bounds/interior gap: {date}")
        sessions.append(
            dict(
                date=date,
                first_hour_start_utc=start.isoformat(),
                last_hour_end_utc=end.isoformat(),
                active_hours=len(rows),
                nominal_end_hour_utc=rule["end_hour_utc"],
                shortened=shortened,
                interior_missing_hours=len(interior),
            )
        )
    sessions = pd.DataFrame(sessions)
    missing_rows = []
    for hour in missing:
        day = sessions[sessions.date == hour.strftime("%Y-%m-%d")]
        inside = bool(
            len(day)
            and pd.Timestamp(day.iloc[0].first_hour_start_utc)
            <= hour
            < pd.Timestamp(day.iloc[0].last_hour_end_utc)
        )
        missing_rows.append(dict(hour_utc=hour.isoformat(), inside_observed_active_session=inside))
    comparison = daily[["date", *OHLC, "volume"]].merge(
        rebuilt.add_suffix("_h1"), left_on="date", right_index=True, how="outer"
    )
    comparison["active_d1"] = comparison.volume > 0
    comparison["active_h1"] = comparison.open_h1.notna()
    comparison["session_match"] = comparison.active_d1 == comparison.active_h1
    differences = []
    for col in OHLC:
        comparison[f"{col}_difference"] = comparison[f"{col}_h1"] - comparison[col]
        differences.append(f"{col}_difference")
    comparison["OHLC_match"] = ~comparison.active_d1 | (comparison[differences].abs() <= 1e-8).all(
        axis=1
    )
    mismatches = comparison[~(comparison.session_match & comparison.OHLC_match)]
    summary = dict(
        status="passed" if not len(mismatches) else "failed",
        hourly_rows=len(frame),
        active_hourly_rows=len(active),
        missing_calendar_hours=len(missing),
        missing_inside_observed_session=sum(
            r["inside_observed_active_session"] for r in missing_rows
        ),
        daily_rows=len(daily),
        active_sessions=len(rebuilt),
        inactive_dates=int((daily.volume == 0).sum()),
        shortened_sessions=int(sessions.shortened.sum()),
        mismatched_dates=len(mismatches),
        max_absolute_OHLC_difference=float(comparison[differences].abs().max().max()),
        assignment_time=settings["assignment_time"],
        cutoff_utc=cutoff.isoformat(),
        last_active_hour_end_utc=active.end.max().isoformat(),
        prices_formed_after_cutoff=len(future),
        D1_is_primary=True,
        model_data_changed=False,
        limitation="Same supplier. Missing hours outside the observed session are listed, not filled.",
    )
    if write:
        out = root / "results/audit"
        out.mkdir(parents=True, exist_ok=True)
        rebuilt.to_csv(out / "h1_rebuilt_daily.csv")
        comparison.to_csv(out / "d1_h1_comparison.csv", index=False)
        mismatches.to_csv(out / "d1_h1_mismatches.csv", index=False)
        sessions.to_csv(out / "daily_sessions.csv", index=False)
        pd.DataFrame(missing_rows, columns=["hour_utc", "inside_observed_active_session"]).to_csv(
            out / "missing_hours.csv", index=False
        )
        (out / "full_h1_review.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if len(mismatches):
        raise ValueError("D1 and H1 disagree; review mismatches before changing the experiment")
    return summary


if __name__ == "__main__":
    print(json.dumps(review(), indent=2))
