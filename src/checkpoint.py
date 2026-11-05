"""Append-only attempts; only a successful attempt completes a forecast key."""

import json
import time


def forecast_key(entry):
    return tuple(entry[k] for k in ["candidate", "seed", "horizon", "origin_index"])


def effective_entries(attempts):
    state = {}
    for entry in attempts:
        key = forecast_key(entry)
        previous = state.get(key)
        if previous and previous["status"] == "ok":
            if entry["status"] == "ok":
                raise ValueError(f"Duplicate successful checkpoint: {key}")
            continue
        state[key] = entry
    return state


def attempt_forecast(stream, state, entry, fit):
    key = forecast_key(entry)
    if key in state and state[key]["status"] == "ok":
        return state[key]
    start = time.perf_counter()
    try:
        prediction, diagnostics = fit()
        entry.update(status="ok", predicted=prediction, diagnostics=diagnostics)
    except Exception as exc:
        entry.update(status="failed", error=repr(exc))
    entry["fit_forecast_seconds"] = time.perf_counter() - start
    stream.write(json.dumps(entry) + "\n")
    stream.flush()
    state[key] = entry
    return entry
