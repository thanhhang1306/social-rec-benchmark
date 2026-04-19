"""Figures for Chapter 6: deployment."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_palette import MODEL_COLORS

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]
rcParams["mathtext.fontset"] = "cm"
rcParams["font.size"] = 13

CSV = Path("/n/fs/recbench/new_rec/results/all_results.csv")
OUT_DIR = Path("/n/fs/recbench/new_rec/figures/deployment")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_DATASETS = [("yelp", "Yelp"), ("lastfm", "Last.fm"), ("douban-book", "Douban-Book")]
MODEL_ORDER = [
    "BPR",
    "LightGCN",
    "SimGCL",
    "DiffNet",
    "MHCN",
    "HAGN",
    "CSGCN",
    "ASLGCN",
]
FAMILY_MARKER = {
    "BPR": "o",
    "LightGCN": "o",
    "SimGCL": "o",
    "DiffNet": "s",
    "MHCN": "s",
    "HAGN": "^",
    "CSGCN": "^",
    "ASLGCN": "^",
}


def load_real_means():
    df = pd.read_csv(CSV)
    real = df[
        (df["batch"] == "TUNED_BATCH_REAL")
        & (df["dataset"].isin([k for k, _ in REAL_DATASETS]))
        & (df["model"].isin(MODEL_ORDER))
    ]
    agg = (
        real.groupby(["dataset", "model"])
        .agg(
            train_time_s=("train_time_s", "mean"),
            inference_time_s=("inference_time_s", "mean"),
            ndcg_10=("ndcg@10", "mean"),
        )
        .reset_index()
    )
    return agg


def _fmt_seconds(v, _):
    if v >= 3600:
        return f"{v / 3600:.0f}h"
    if v >= 60:
        return f"{v / 60:.0f}m"
    return f"{v:.0f}s"


def fig_cost_vs_accuracy(agg):
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (key, label) in zip(axes, REAL_DATASETS):
        sub = agg[agg["dataset"] == key].set_index("model").reindex(MODEL_ORDER)
        for m in MODEL_ORDER:
            row = sub.loc[m]
            ax.scatter(
                row["train_time_s"],
                row["ndcg_10"],
                color=MODEL_COLORS[m],
                marker=FAMILY_MARKER[m],
                s=140,
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
                label=m if ax is axes[0] else None,
            )

        ax.set_xscale("log")
        all_times = sub["train_time_s"].values
        tick_candidates = [10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200]
        tmin, tmax = all_times.min() * 0.7, all_times.max() * 1.4
        ticks = [t for t in tick_candidates if tmin <= t <= tmax]
        if len(ticks) < 3:
            ticks = [tick_candidates[0]] + ticks + [tick_candidates[-1]]
            ticks = sorted(set(t for t in ticks if tmin * 0.5 <= t <= tmax * 2))
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_seconds))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(tmin, tmax)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Training time (log scale)")
        ax.grid(alpha=0.3, which="major")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("NDCG@10 (tuned)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=len(MODEL_ORDER),
        fontsize=10,
        framealpha=0.9,
        edgecolor="none",
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"train_cost_vs_accuracy.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote train_cost_vs_accuracy.{pdf,png}")


def main():
    agg = load_real_means()
    fig_cost_vs_accuracy(agg)


if __name__ == "__main__":
    main()
