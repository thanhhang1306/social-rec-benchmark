"""Figure for §4.7: paired default versus tuned NDCG@10 on real-world datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]
rcParams["mathtext.fontset"] = "cm"
rcParams["font.size"] = 14

RESULTS_ROOT = Path("/n/fs/recbench/new_rec/results")
OUT_DIR = Path("/n/fs/recbench/new_rec/figures/results")

MODELS = ["BPR", "DiffNet", "LightGCN", "MHCN", "SimGCL"]
DATASETS = [("yelp", "Yelp"), ("douban-book", "Douban Book"), ("lastfm", "Last.fm")]
MSEEDS = ["mseed_1", "mseed_12", "mseed_123"]

MODEL_COLORS = {
    "BPR": "#7F7F7F",
    "DiffNet": "#D55E00",
    "LightGCN": "#0072B2",
    "MHCN": "#009E73",
    "SimGCL": "#CC79A7",
}


def load_ndcg(batch: str, mseed: str, dataset: str, model: str) -> float | None:
    path = RESULTS_ROOT / batch / mseed / dataset / model / "final_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "ndcg@10" not in df.columns:
        return None
    return float(df["ndcg@10"].iloc[0])


def collect() -> pd.DataFrame:
    rows = []
    for dataset_key, dataset_label in DATASETS:
        for model in MODELS:
            for mseed in MSEEDS:
                for batch, condition in [
                    ("DEFAULT_BATCH_REAL", "default"),
                    ("TUNED_BATCH_REAL", "tuned"),
                ]:
                    val = load_ndcg(batch, mseed, dataset_key, model)
                    if val is None:
                        continue
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "model": model,
                            "mseed": mseed,
                            "condition": condition,
                            "ndcg": val,
                        }
                    )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=False)

    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset_label]
        means = sub.groupby(["model", "condition"])["ndcg"].mean().unstack("condition")
        means = means.reindex(MODELS)

        x_default = 0
        x_tuned = 1

        for model in MODELS:
            if model not in means.index:
                continue
            d = means.loc[model, "default"]
            t = means.loc[model, "tuned"]
            if pd.isna(d) or pd.isna(t):
                continue
            color = MODEL_COLORS[model]
            ax.plot(
                [x_default, x_tuned],
                [d, t],
                "-o",
                color=color,
                linewidth=2,
                markersize=6,
                label=model,
            )

        ax.set_xticks([x_default, x_tuned])
        ax.set_xticklabels(["Default", "Tuned"], fontsize=14)
        ax.set_xlim(-0.15, 1.15)
        ax.set_xlabel(dataset_label, fontsize=15)
        ax.set_ylabel("NDCG@10" if ax is axes[0] else "", fontsize=15)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([], [], color=MODEL_COLORS[m], marker="o", linewidth=2, markersize=6, label=m)
        for m in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(MODELS),
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"tuning_crossings_real.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_DIR}/tuning_crossings_real.{{pdf,png}}")


def main() -> None:
    df = collect()
    print(df.groupby(["dataset", "model", "condition"])["ndcg"].mean().round(4))
    plot(df)


if __name__ == "__main__":
    main()
