import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_data import hourly_frame, rebuild_daily, review


def candle(hour, price=100, volume=1):
    return dict(
        timestamp=int(pd.Timestamp(f"2026-10-27T{hour:02d}:00:00Z").timestamp() * 1000),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=volume,
    )


class HourlyReviewTests(unittest.TestCase):
    def test_aggregation_uses_open_high_low_close_in_order(self):
        frame = hourly_frame([candle(19, 100), candle(20, 110)])
        row = rebuild_daily(frame, "2026-10-27T21:00:00Z").iloc[0]
        self.assertEqual(row.to_dict(), dict(open=100, high=111, low=99, close=110))

    def test_incomplete_future_hour_does_not_enter_daily_close(self):
        frame = hourly_frame([candle(20, 100), candle(21, 200)])
        row = rebuild_daily(frame, "2026-10-27T21:00:00Z").iloc[0]
        self.assertEqual(row.close, 100)

    def test_duplicates_and_invalid_hour_labels_rejected(self):
        with self.assertRaises(ValueError):
            hourly_frame([candle(20), candle(20)])
        row = candle(20)
        row["timestamp"] += 1000
        with self.assertRaises(ValueError):
            hourly_frame([row])

    def test_bad_prices_and_moving_zero_volume_rejected(self):
        for field, value in [("close", float("nan")), ("low", 0), ("volume", -1), ("volume", 0)]:
            row = candle(20)
            row[field] = value
            with self.assertRaises(ValueError):
                hourly_frame([row])

    def fixture(self, root, records):
        (root / "config").mkdir()
        (root / "data/raw").mkdir(parents=True)
        config = dict(
            assignment_time="2026-10-28T00:00:00+03:00",
            cutoff_utc="2026-10-27T21:00:00+00:00",
            from_utc="2026-10-27T00:00:00+00:00",
            to_utc_exclusive="2026-10-28T00:00:00+00:00",
            nominal_session_changes_observed_in_archive=[
                {"from": "2026-10-27", "start_hour_utc": 10, "end_hour_utc": 21}
            ],
        )
        (root / "config/data_review.json").write_text(json.dumps(config))
        source = root / "data/raw/brent_h1_bid_2025-10-28_2026-10-28.json"
        source.write_text(json.dumps(records))
        (root / "data/raw/h1_metadata.json").write_text(
            json.dumps(
                dict(rows=len(records), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            )
        )
        (root / "data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv").write_text(
            "timestamp_utc,open,high,low,close,volume\n2026-10-27T00:00:00.000Z,100,101,99,100,1\n"
        )

    def test_missing_interior_hour_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, [candle(10), candle(12)])
            with self.assertRaisesRegex(ValueError, "interior gap"):
                review(root, write=False)

    def test_active_prices_after_assignment_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root, [candle(20), candle(21)])
            with self.assertRaisesRegex(ValueError, "after assignment"):
                review(root, write=False)

    def test_complete_saved_year_matches_D1_and_cutoff(self):
        report = review(write=False)
        self.assertEqual(report["active_sessions"], 258)
        self.assertEqual(report["mismatched_dates"], 0)
        self.assertEqual(report["max_absolute_OHLC_difference"], 0)
        self.assertEqual(report["missing_inside_observed_session"], 0)
        self.assertEqual(report["last_active_hour_end_utc"], report["cutoff_utc"])
        self.assertEqual(
            pd.Timestamp(report["assignment_time"]), pd.Timestamp("2026-10-27T21:00:00Z")
        )


if __name__ == "__main__":
    unittest.main()
