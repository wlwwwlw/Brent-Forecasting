"""Recalculate artifacts and diagnostics without fitting or selecting new models."""

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from checkpoint import effective_entries
from experiment import ROOT, metrics
from integrity import verify_files
from prepare_data import review

OUT = ROOT / "results/audit"
FIG = ROOT / "results/figures"
NAMES = {
    "naive": "Naive",
    "ses": "SES",
    "elastic_1.0": "ElasticNet",
    "lstm_8": "LSTM",
    "rnn_8": "RNN",
}
COLORS = {
    "naive": "#65788c",
    "ses": "#d5a24a",
    "elastic_1.0": "#318b80",
    "lstm_8": "#8864af",
    "rnn_8": "#8864af",
}


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def verify_predictions(prices, dates):
    checks = []
    files = [
        ("baseline", "baseline_metrics.csv"),
        ("statistical", "statistical_metrics.csv"),
        ("ml", "ml_metrics.csv"),
        ("neural", "neural_seed_metrics.csv"),
    ]
    expected_origins = list(range(99, 203))
    for name, score_file in files + [("test", "metrics.csv")]:
        directory = ROOT / ("results/test" if name == "test" else "results/validation")
        frame = pd.read_csv(
            directory / ("predictions.csv" if name == "test" else f"{name}_predictions.csv")
        )
        origin = frame.origin_index.to_numpy()
        target = frame.target_index.to_numpy()
        np.testing.assert_allclose(frame.current, prices[origin], atol=1e-12, rtol=0)
        np.testing.assert_allclose(frame.actual, prices[target], atol=1e-12, rtol=0)
        assert (frame.target_index == frame.origin_index + frame.horizon).all()
        assert (frame.origin_date.to_numpy() == dates[origin]).all()
        assert (frame.target_date.to_numpy() == dates[target]).all()
        assert np.isfinite(frame.predicted).all()
        assert (frame.predicted > 0).all()
        keys = ["candidate", "horizon"] + (["seed"] if "seed" in frame else [])
        for _, g in frame.groupby(keys):
            assert sorted(g.origin_index.tolist()) == (
                list(range(207, 253)) if name == "test" else expected_origins
            )
        scores = pd.read_csv(directory / score_file)
        for _, score in scores.iterrows():
            sub = frame[(frame.candidate == score.candidate) & (frame.horizon == score.horizon)]
            if "seed" in score.index:
                sub = sub[sub.seed == score.seed]
            if name != "test" and score.fold != "all":
                sub = sub[sub.fold == score.fold]
            if name == "test":
                if score.period == "first_half":
                    sub = sub[sub.origin_index <= 229]
                elif score.period == "second_half":
                    sub = sub[sub.origin_index >= 230]
                elif score.period == "without_last_origin":
                    sub = sub[sub.origin_index < 252]
            actual = metrics(sub.actual.tolist(), sub.predicted.tolist(), sub.current.tolist())
            for key, value in actual.items():
                np.testing.assert_allclose(
                    value, score[key], atol=1e-10, rtol=0, err_msg=f"{name}/{key}"
                )
        checks.append(
            dict(dataset=name, forecasts=len(frame), metric_rows=len(scores), status="passed")
        )
    return checks


def verify_training(prices):
    fits = read_json("results/validation/ml_fit_diagnostics.json")
    attempts = [
        json.loads(line)
        for line in (ROOT / "results/validation/neural_checkpoint.jsonl").read_text().splitlines()
        if line
    ]
    neural = list(effective_entries(attempts).values())
    final = read_json("results/test/diagnostics.json")
    entries = [(r, r) for r in fits] + [(r, r["diagnostics"]) for r in neural]
    entries += [(r, r) for r in final if r["family"] in ["ml", "neural"]]
    cache = {}
    for row, diag in entries:
        t, h = row["origin_index"], row["horizon"]
        assert diag["last_training_target"] == t
        assert diag["last_training_origin"] + h == t
        assert diag["training_n"] == t - h - 19
        if (t, h) not in cache:
            origins = np.arange(20, t - h + 1)
            delta = prices[origins + h] - prices[origins]
            returns = prices[1 : t + 1] / prices[:t] - 1
            windows = np.stack([returns[j - 20 : j] for j in origins])
            cache[t, h] = (delta.mean(), delta.std(), windows.mean(), windows.std())
        ym, ys, xm, xs = cache[t, h]
        if "scaler_n" in diag:
            assert diag["scaler_n"] == diag["training_n"]
            np.testing.assert_allclose(
                [diag["target_mean"], diag["target_scale"]], [ym, ys], atol=1e-10, rtol=0
            )
        else:
            np.testing.assert_allclose(
                [diag["y_mean"], diag["y_scale"], diag["x_mean"], diag["x_scale"]],
                [ym, ys, xm, xs],
                atol=1e-10,
                rtol=0,
            )
    test_predictions = pd.read_csv(ROOT / "results/test/predictions.csv")
    for row in final:
        if row["family"] != "ml":
            continue
        match = test_predictions[
            (test_predictions.family == "ml")
            & (test_predictions.horizon == row["horizon"])
            & (test_predictions.origin_index == row["origin_index"])
        ]
        np.testing.assert_allclose(
            match.predicted.iloc[0] - match.current.iloc[0], row["target_mean"], atol=1e-9, rtol=0
        )
    return len(entries)


def data_diagnostics(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data.timestamp_utc, utc=True)
    data["return"] = data.close.pct_change()
    data["gap"] = data.open / data.close.shift() - 1
    data["abs_return"] = data["return"].abs()
    data[["close", "return", "gap"]].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).to_csv(OUT / "descriptive_statistics.csv")
    top = data.nlargest(10, "abs_return").copy()
    daily = pd.read_csv(ROOT / "data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv").set_index(
        "timestamp_utc"
    )
    hourly = read_json("data/validation/anomaly_hourly.json")
    checks = []
    for entry in hourly:
        candles = sorted(
            [r for r in entry["candles"] if r["volume"] > 0], key=lambda r: r["timestamp"]
        )
        row = daily.loc[entry["date"] + "T00:00:00.000Z"]
        aggregate = (
            [
                candles[0]["open"],
                max(r["high"] for r in candles),
                min(r["low"] for r in candles),
                candles[-1]["close"],
            ]
            if candles
            else [row.close] * 4
        )
        np.testing.assert_allclose(
            row[["open", "high", "low", "close"]].astype(float), aggregate, atol=1e-8, rtol=0
        )
        assert bool(candles) == (row.volume > 0)
        checks.append(dict(date=entry["date"], active_hours=len(candles), OHLC_match=True))
    pd.DataFrame(checks).to_csv(OUT / "anomaly_hourly_checks.csv", index=False)
    top["hourly_OHLC_match"] = True
    top[
        [
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return",
            "gap",
            "hourly_OHLC_match",
        ]
    ].to_csv(OUT / "largest_moves.csv", index=False)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].plot(data.date, data.close, color="#263e55", lw=1.4)
    axes[0].axvspan(
        data.date.iloc[99],
        data.date.iloc[207],
        alpha=0.14,
        color="#d5a24a",
        label="Выбор моделей",
    )
    axes[0].axvspan(
        data.date.iloc[208],
        data.date.iloc[-1],
        alpha=0.14,
        color="#8864af",
        label="Тестовые цели",
    )
    axes[0].set_ylabel("Цена, USD/баррель")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("Brent: цена закрытия и её изменение")
    axes[1].bar(data.date, 100 * data["return"], color="#318b80", width=1)
    axes[1].set_ylabel("Изменение цены, %")
    save(fig, "market_and_returns.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(data.close, bins=20, color="#65788c")
    axes[0].set_xlabel("Цена, USD/баррель")
    axes[1].hist(100 * data["return"].dropna(), bins=25, color="#318b80")
    axes[1].set_xlabel("Изменение цены, %")
    for ax in axes:
        ax.set_ylabel("Сессии")
    save(fig, "data_distributions.png")
    return checks


def final_diagnostics():
    df = pd.read_csv(ROOT / "results/test/predictions.csv")
    df = df[df.seed == 42].copy()
    df["error"] = df.predicted - df.actual
    df["abs_error"] = df.error.abs()
    directions = []
    influence = []
    costs = []
    diagnostics = pd.DataFrame(read_json("results/test/diagnostics.json"))
    for (cid, h), g in df.groupby(["candidate", "horizon"]):
        base = df[(df.candidate == "naive") & (df.horizon == h)].sort_values("origin_index")
        g = g.sort_values("origin_index")
        sign = np.sign(g.actual - g.current)
        observed = np.sign(g.predicted - g.current)
        rmse = np.sqrt(np.mean(g.error**2))
        base_rmse = np.sqrt(np.mean(base.error**2))
        nonzero = observed != 0
        directions.append(
            dict(
                candidate=cid,
                horizon=h,
                RMSE=rmse,
                RMSE_gain_pct=100 * (1 - rmse / base_rmse),
                direction_accuracy_pct=100 * np.mean(observed == sign),
                coverage_pct=100 * np.mean(nonzero),
                conditional_direction_accuracy_pct=float(
                    100 * np.mean(observed[nonzero] == sign[nonzero])
                )
                if nonzero.any()
                else None,
                always_up_pct=100 * np.mean(sign == 1),
                always_down_pct=100 * np.mean(sign == -1),
                mean_error=g.error.mean(),
                p95_absolute_error=g.abs_error.quantile(0.95),
                max_absolute_error=g.abs_error.max(),
            )
        )
        for i in range(len(g)):
            keep = np.arange(len(g)) != i
            gain = np.sqrt(np.mean(g.error.to_numpy()[keep] ** 2)) - np.sqrt(
                np.mean(base.error.to_numpy()[keep] ** 2)
            )
            influence.append(
                dict(
                    candidate=cid,
                    horizon=h,
                    removed_origin=int(g.origin_index.iloc[i]),
                    RMSE_difference_without_origin=gain,
                )
            )
        timing = diagnostics[
            (diagnostics.candidate == cid) & (diagnostics.horizon == h) & (diagnostics.seed == 42)
        ]
        costs.append(
            dict(
                candidate=cid,
                horizon=h,
                fits=len(timing),
                median_ms=1000 * timing.fit_forecast_seconds.median(),
                total_seconds=timing.fit_forecast_seconds.sum(),
                RMSE=rmse,
            )
        )
    pd.DataFrame(directions).to_csv(OUT / "direction_and_errors.csv", index=False)
    pd.DataFrame(influence).to_csv(OUT / "leave_one_origin_out.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "final_runtime.csv", index=False)
    largest = df.sort_values("abs_error", ascending=False).groupby(["candidate", "horizon"]).head(3)
    largest.to_csv(OUT / "largest_forecast_errors.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, h in zip(axes, [1, 5]):
        sub = df[df.horizon == h]
        truth = sub[sub.candidate == "naive"].sort_values("target_date")
        ax.plot(
            pd.to_datetime(truth.target_date), truth.actual, color="#182c40", lw=2, label="Факт"
        )
        for cid, g in sub.groupby("candidate"):
            g = g.sort_values("target_date")
            ax.plot(
                pd.to_datetime(g.target_date),
                g.predicted,
                lw=1,
                alpha=0.85,
                color=COLORS[cid],
                label=NAMES[cid],
            )
        ax.set_title(f"Горизонт {h}: факт и прогноз по датам цели (seed=42)")
        ax.set_ylabel("USD/баррель")
        ax.legend(ncol=5, fontsize=8)
    save(fig, "test_forecasts.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, h in enumerate([1, 5]):
        sub = df[df.horizon == h]
        for cid, g in sub.groupby("candidate"):
            axes[row, 0].hist(
                g.error, bins=12, histtype="step", lw=1.5, color=COLORS[cid], label=NAMES[cid]
            )
        axes[row, 0].axvline(0, color="black", lw=0.5)
        axes[row, 0].set_xlabel("Прогноз минус факт, USD/баррель")
        axes[row, 0].set_title(f"Горизонт {h}: ошибки со знаком")
        axes[row, 0].legend(fontsize=8)
        groups = list(sub.groupby("candidate"))
        axes[row, 1].boxplot(
            [g.abs_error for _, g in groups], tick_labels=[NAMES[c] for c, _ in groups]
        )
        axes[row, 1].set_title(f"Горизонт {h}: абсолютные ошибки")
        axes[row, 1].set_ylabel("USD/баррель")
    save(fig, "test_error_distributions.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, h in zip(axes, [1, 5]):
        for cid, g in df[df.horizon == h].groupby("candidate"):
            ax.scatter(
                abs(g.actual - g.current),
                g.abs_error,
                s=16,
                alpha=0.65,
                label=NAMES[cid],
                color=COLORS[cid],
            )
        ax.set_xlabel("Абсолютное изменение цены")
        ax.set_ylabel("Абсолютная ошибка прогноза")
        ax.set_title(f"Горизонт {h}, USD/баррель")
        ax.legend(fontsize=8)
    save(fig, "errors_vs_price_changes.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h in zip(axes, [1, 5]):
        for r in costs:
            if r["horizon"] != h:
                continue
            ax.scatter(max(r["median_ms"], 0.001), r["RMSE"], color=COLORS[r["candidate"]], s=50)
            ax.annotate(
                NAMES[r["candidate"]],
                (max(r["median_ms"], 0.001), r["RMSE"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Подготовка, обучение и прогноз: медиана, мс (лог. шкала)")
        ax.set_ylabel("RMSE на тесте")
        ax.set_title(f"Горизонт {h}: одна машина, seed=42")
    save(fig, "quality_vs_cost.png")


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    FIG.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(ROOT / "data/processed/brent_active_provisional.csv")
    checks = verify_predictions(data.close.to_numpy(), data.timestamp_utc.str[:10].to_numpy())
    training = verify_training(data.close.to_numpy())
    lock = read_json("config/final_lock.json")
    verify_files(lock["hashes"])
    original = read_json("provenance/v1_artifact_hashes.json")
    verify_files(original)
    hourly = data_diagnostics(data)
    final_diagnostics()
    summary = dict(
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        status="passed",
        predictions=checks,
        training_diagnostics_checked=training,
        extended_hourly_dates=len(hourly),
        lock_hashes_verified=len(lock["hashes"]),
        test_reselected=False,
        original_artifact_hashes_verified=len(original),
        full_h1_review=review(write=False),
        limitations=[
            "Same supplier cross-frequency validation; no independent price feed.",
            "Diagnostic analysis after test; not new independent evidence.",
        ],
    )
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
