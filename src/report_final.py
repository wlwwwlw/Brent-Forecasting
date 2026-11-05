"""Paired final metrics, seed sensitivity and exploratory dependent-error intervals."""

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import ROOT, metrics, write_csv


def bootstrap_difference(actual, predicted, baseline, block, replicates=10000, seed=20261104):
    a, p, b = [np.asarray(x, dtype=float) for x in (actual, predicted, baseline)]
    if a.ndim != 1 or not len(a) or a.shape != p.shape or a.shape != b.shape:
        raise ValueError("Aligned nonempty vectors required")
    if not all(np.isfinite(x).all() for x in (a, p, b)) or not 1 <= block <= len(a):
        raise ValueError("Invalid bootstrap inputs")
    rng = np.random.default_rng(seed)
    n = len(a)
    starts = rng.integers(0, n, size=(replicates, (n + block - 1) // block))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(replicates, -1)[:, :n]
    error = (p - a)[indices]
    reference = (b - a)[indices]
    delta = np.sqrt(np.mean(error**2, axis=1)) - np.sqrt(np.mean(reference**2, axis=1))
    return float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))


def main():
    out = ROOT / "results/test"
    lock = json.loads((ROOT / "config/final_lock.json").read_text())
    df = pd.read_csv(out / "predictions.csv")
    summaries = []
    for (family, candidate, h, seed), group in df.groupby(
        ["family", "candidate", "horizon", "seed"]
    ):
        group = group.sort_values("origin_index")
        if group.origin_index.tolist() != list(range(207, 253)):
            raise AssertionError("Unmatched origins")
        for label, subset in [
            ("all", group),
            ("first_half", group[group.origin_index <= 229]),
            ("second_half", group[group.origin_index >= 230]),
            ("without_last_origin", group[group.origin_index < 252]),
        ]:
            score = metrics(
                subset.actual.tolist(), subset.predicted.tolist(), subset.current.tolist()
            )
            summaries.append(
                dict(
                    family=family,
                    candidate=candidate,
                    horizon=int(h),
                    seed=int(seed),
                    period=label,
                    **score,
                )
            )
    write_csv(out / "metrics.csv", summaries)
    scores = pd.DataFrame(summaries)
    primary = scores[(scores.seed == 42) & (scores.period == "all")]
    primary.to_csv(out / "primary_metrics.csv", index=False)
    neural = scores[(scores.family == "neural") & (scores.period == "all")]
    neural.groupby(["candidate", "horizon"])[
        ["RMSE", "MAE", "sMAPE_pct", "direction_accuracy_pct"]
    ].agg(["mean", "std", "min", "max"]).to_csv(out / "neural_seed_summary.csv")
    intervals = []
    for h in [1, 5]:
        base = df[(df.family == "simple") & (df.horizon == h)].sort_values("origin_index")
        for family in ["statistical", "ml", "neural"]:
            group = df[(df.family == family) & (df.horizon == h) & (df.seed == 42)].sort_values(
                "origin_index"
            )
            if not np.array_equal(group.actual, base.actual):
                raise AssertionError("Mismatched targets")
            for block in lock["bootstrap"]["block_lengths"]:
                lo, hi = bootstrap_difference(
                    group.actual,
                    group.predicted,
                    base.predicted,
                    block,
                    lock["bootstrap"]["replicates"],
                    lock["bootstrap"]["seed"],
                )
                difference = float(
                    np.sqrt(np.mean((group.predicted - group.actual) ** 2))
                    - np.sqrt(np.mean((base.predicted - base.actual) ** 2))
                )
                intervals.append(
                    dict(
                        family=family,
                        candidate=group.candidate.iloc[0],
                        horizon=h,
                        seed=42,
                        block_length=block,
                        RMSE_difference=difference,
                        lower_95=lo,
                        upper_95=hi,
                    )
                )
    write_csv(out / "paired_bootstrap.csv", intervals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, h in zip(axes, [1, 5]):
        sub = primary[primary.horizon == h].sort_values("RMSE")
        ax.barh(
            sub.candidate,
            sub.RMSE,
            color=["#5c708a" if x == "naive" else "#8564ab" for x in sub.candidate],
        )
        ax.invert_yaxis()
        ax.set_title(f"Horizon {h} trading day(s)")
        ax.set_xlabel("Test RMSE, USD/barrel")
        ax.set_xlim(0, sub.RMSE.max() * 1.2)
        for i, v in enumerate(sub.RMSE):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center")
    fig.suptitle("Locked finalists: 46 test origins; neural reference seed 42")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/final_test_rmse.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h in zip(axes, [1, 5]):
        candidates = ["ses", "elastic_1.0", lock["winners"]["neural"][str(h)]]
        for block, color, offset in [(5, "#318b80", -0.10), (10, "#8864af", 0.10)]:
            for i, cid in enumerate(candidates):
                row = next(
                    r
                    for r in intervals
                    if r["horizon"] == h and r["candidate"] == cid and r["block_length"] == block
                )
                ax.plot([row["lower_95"], row["upper_95"]], [i + offset] * 2, color=color, lw=2)
                ax.plot(
                    row["RMSE_difference"],
                    i + offset,
                    "o",
                    color=color,
                    label=f"Блок {block}" if i == 0 else None,
                )
        ax.axvline(0, color="#444444", ls="--", lw=1)
        ax.set_yticks(range(3), ["SES", "ElasticNet", "LSTM" if h == 1 else "RNN"])
        ax.set_title(f"Горизонт {h}: разница RMSE с Naive")
        ax.set_xlabel("USD/баррель; левее нуля — меньше ошибка")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/rmse_difference_intervals.png", dpi=170)
    plt.close(fig)
    print(primary[["candidate", "horizon", "RMSE", "MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
