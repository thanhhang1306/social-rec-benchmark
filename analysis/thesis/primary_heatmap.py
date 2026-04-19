"""Figure for §4.3: topology-by-model NDCG@10 heatmap at the primary condition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

BASE = Path("/n/fs/recbench/new_rec/results/TUNED_BATCH_SYNTH")
OUT_DIR = Path("/n/fs/recbench/new_rec/figures/results")

MODELS = ["SimGCL", "MHCN", "LightGCN", "DiffNet", "BPR"]
DSEEDS = ["dseed_1", "dseed_2", "dseed_3"]
MSEEDS = ["mseed_1", "mseed_12", "mseed_123"]

TIER_BOUNDARIES = [0.5, 1.5, 2.5]
TIER_LABELS = [
    ("SSL-\nContrastive", 0),
    ("SSL-\nHypergraph", 1),
    ("GCN", 2),
    ("Bottom", 3.5),
]

TOPOLOGIES = [
    ("scale_free", "Scale-Free", "I"),
    ("small_world", "Small-World", "I"),
    ("random", "Random", "I"),
    ("echo_chamber", "Echo Chamber", "II"),
    ("partial_alignment", "Partial Alignment", "II"),
    ("contrarian", "Contrarian", "II"),
    ("star", "Star", "III"),
    ("line_graph", "Line Graph", "III"),
    ("fake_users", "Fake Users", "III"),
]

GROUP_BOUNDARIES = [2.5, 5.5]


def config_dir(topo: str) -> str:
    if topo == "fake_users":
        return "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.1"
    if topo == "line_graph":
        return "line_graph_u2000_i5000_sp0.01_ns0.1_max30_decay0.5"
    return f"{topo}_u2000_i5000_sp0.01_ns0.1"


def load_cell(topo: str, model: str) -> float:
    vals = []
    for dseed in DSEEDS:
        for mseed in MSEEDS:
            path = BASE / dseed / mseed / config_dir(topo) / model / "final_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty or "ndcg@10" not in df.columns:
                continue
            vals.append(float(df["ndcg@10"].iloc[0]))
    return float(np.mean(vals)) if vals else np.nan


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 12,
        }
    )

    matrix = np.zeros((len(TOPOLOGIES), len(MODELS)))
    for i, (topo, _, _) in enumerate(TOPOLOGIES):
        for j, model in enumerate(MODELS):
            matrix[i, j] = load_cell(topo, model)

    vmin = max(np.nanmin(matrix[matrix > 0]), 1e-4)
    vmax = np.nanmax(matrix)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        matrix,
        cmap="viridis",
        aspect="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=12)
    ax.set_yticks(range(len(TOPOLOGIES)))
    ax.set_yticklabels([t[1] for t in TOPOLOGIES], fontsize=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            color = "white" if v < np.sqrt(vmin * vmax) * 3 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=color)

    for x in TIER_BOUNDARIES:
        ax.axvline(x=x, color="white", linewidth=2.5)

    for y in GROUP_BOUNDARIES:
        ax.axhline(y=y, color="black", linewidth=1.5, linestyle="--")

    for label, xpos in TIER_LABELS:
        ax.text(xpos, -1.1, label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.text(
        -2.3,
        1.0,
        "Group I",
        fontsize=10,
        rotation=90,
        va="center",
        ha="center",
        color="#555",
        style="italic",
    )
    ax.text(
        -2.3,
        4.0,
        "Group II",
        fontsize=10,
        rotation=90,
        va="center",
        ha="center",
        color="#555",
        style="italic",
    )
    ax.text(
        -2.3,
        7.0,
        "Group III",
        fontsize=10,
        rotation=90,
        va="center",
        ha="center",
        color="#555",
        style="italic",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("NDCG@10 (log scale)", fontsize=11)

    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(len(TOPOLOGIES) - 0.5, -0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"primary_heatmap.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_DIR}/primary_heatmap.{{pdf,png}}")

    print("\nValues:")
    print(pd.DataFrame(matrix, index=[t[1] for t in TOPOLOGIES], columns=MODELS).round(4))


if __name__ == "__main__":
    main()
