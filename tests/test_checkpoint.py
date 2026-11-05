import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from checkpoint import attempt_forecast, effective_entries


class CheckpointTests(unittest.TestCase):
    def test_failed_fit_retried_after_reload_once_then_skipped(self):
        entry = dict(candidate="rnn_8", seed=42, horizon=5, origin_index=99)
        stream = io.StringIO()
        state = {}

        def fail():
            raise RuntimeError("intentional test failure")

        attempt_forecast(stream, state, dict(entry), fail)
        state = effective_entries([json.loads(line) for line in stream.getvalue().splitlines()])
        self.assertEqual(next(iter(state.values()))["status"], "failed")
        calls = []

        def succeed():
            calls.append(1)
            return 100.0, {"training_n": 75}

        attempt_forecast(stream, state, dict(entry), succeed)
        attempt_forecast(stream, state, dict(entry), succeed)
        attempts = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(attempts), 2)
        effective = effective_entries(attempts)
        self.assertEqual(len(effective), 1)
        self.assertEqual(next(iter(effective.values()))["status"], "ok")

    def test_duplicate_success_rejected(self):
        row = dict(candidate="x", seed=1, horizon=1, origin_index=5, status="ok")
        with self.assertRaises(ValueError):
            effective_entries([row, row])
