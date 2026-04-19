"""Figures for Chapter 5: hybrids and ablations."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_palette import MODEL_COLORS, MUTED, SEQUENTIAL_CMAP

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]
rcParams["mathtext.fontset"] = "cm"
rcParams["font.size"] = 13

CSV = Path("/n/fs/recbench/new_rec/results/all_results.csv")
OUT_DIR = Path("/n/fs/recbench/new_rec/figures/hybrids")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HYBRIDS = ["HAGN", "CSGCN", "ASLGCN"]
REAL_DATASETS = [("yelp", "Yelp"), ("lastfm", "Last.fm"), ("douban-book", "Douban-Book")]

FAMILY_RE = re.compile(r"^([a-z_]+?)_u\d")

TOPO_ORDER = [
    "scale_free",
    "small_world",
    "random",
    "echo_chamber",
    "partial_alignment",
    "contrarian",
    "star",
    "line_graph",
    "fake_users",
]
TOPO_PRETTY = {
    "scale_free": "Scale-Free",
    "small_world": "Small-World",
    "random": "Random",
    "echo_chamber": "Echo Ch.",
    "partial_alignment": "Partial Al.",
    "contrarian": "Contrarian",
    "star": "Star",
    "line_graph": "Line Graph",
    "fake_users": "Fake Users",
}

ABLATION_GROUPS = [
    ("HAGN", ["HAGNUniform", "HAGNFixed", "HAGNGlobal"]),
    ("CSGCN", ["CSGCNNoWarmup"]),
    ("ASLGCN", ["ASLGCNSymmetric"]),
]
ABL_PRETTY = {
    "HAGNUniform": "HAGNUniform",
    "HAGNFixed": "HAGNFixed",
    "HAGNGlobal": "HAGNGlobal",
    "CSGCNNoWarmup": "CSGCNNoWarmup",
    "ASLGCNSymmetric": "ASLGCNSymmetric",
}


def load():
    df = pd.read_csv(CSV)
    df["family"] = df["dataset"].str.extract(FAMILY_RE)
    return df


def _parse_synth(ds):
    m = re.match(r"([a-z_]+?)_u(\d+)_i(\d+)_sp([\d.]+)_ns([\d.]+)(?:_deg([\d.]+))?", ds)
    if not m:
        return None
    return {
        "topo": m.group(1),
        "u": int(m.group(2)),
        "i": int(m.group(3)),
        "sp": float(m.group(4)),
        "ns": float(m.group(5)),
        "deg": float(m.group(6)) if m.group(6) else None,
    }


def _classify_sweep(p):
    if p is None or p["deg"] is not None:
        return None
    if p["u"] == 2000 and p["i"] == 5000 and p["ns"] == 0.1:
        return "density"
    if p["u"] == 2000 and p["i"] == 5000 and p["sp"] == 0.01:
        return "noise"
    if p["sp"] == 0.01 and p["ns"] == 0.1:
        return "scale"
    return None


def _sweep_x(sweep, p):
    if sweep == "density":
        return p["sp"]
    if sweep == "noise":
        return p["ns"]
    if sweep == "scale":
        return p["u"] * p["i"]
    return None


def fig_hybrid_sweep_gaps(df):
    models = ["LightGCN"] + HYBRIDS
    sub = df[(df["batch"] == "TUNED_BATCH_SYNTH") & (df["model"].isin(models))].copy()

    params = sub["dataset"].map(_parse_synth)
    sub = sub[params.notna()].copy()
    params = params[params.notna()]
    sub["sweep"] = params.map(_classify_sweep)
    sub["sweep_x"] = [_sweep_x(sw, p) for sw, p in zip(sub["sweep"], params)]
    sub["topo"] = params.map(lambda p: p["topo"])
    sub = sub[sub["sweep"].notna()]

    sweep_info = {
        "density": ("Social Density (sp)", None),
        "noise": ("Noise Level (ns)", None),
        "scale": ("Scale (users × items)", "log"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, sweep_name in zip(axes, ["density", "noise", "scale"]):
        sw = sub[sub["sweep"] == sweep_name]
        xlabel, xscale = sweep_info[sweep_name]

        for hybrid in HYBRIDS:
            lgcn = sw[sw["model"] == "LightGCN"].groupby(["topo", "sweep_x"])["ndcg@10"].mean()
            hyb = sw[sw["model"] == hybrid].groupby(["topo", "sweep_x"])["ndcg@10"].mean()

            delta = hyb - lgcn
            delta = delta.dropna()
            if delta.empty:
                continue

            topo_delta = delta.reset_index()
            topo_delta.columns = ["topo", "sweep_x", "delta"]
            agg = topo_delta.groupby("sweep_x")["delta"].agg(["mean", "std"])
            agg = agg.sort_index()

            xs = agg.index.values
            ym = agg["mean"].values
            ys = agg["std"].values

            ax.plot(
                xs, ym, marker="o", ms=5, color=MODEL_COLORS[hybrid], label=hybrid, lw=1.8, zorder=3
            )
            ax.fill_between(xs, ym - ys, ym + ys, color=MODEL_COLORS[hybrid], alpha=0.15, zorder=1)

        ax.axhline(0, color=MUTED, ls="--", lw=0.8, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_title(sweep_name.capitalize() + " Sweep")
        ax.grid(alpha=0.25)
        if xscale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda v, _: f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"
                )
            )

    axes[0].set_ylabel(r"$\Delta$ NDCG@10 vs LightGCN")
    axes[0].legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"hybrid_sweep_gaps.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote hybrid_sweep_gaps.{pdf,png}")


def fig_real_hybrid_seeds(df):
    models = ["LightGCN"] + HYBRIDS
    sub = df[
        (df["batch"] == "TUNED_BATCH_REAL")
        & (df["dataset"].isin([k for k, _ in REAL_DATASETS]))
        & (df["model"].isin(models))
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), sharey=False)

    for ax, (ds_key, ds_label) in zip(axes, REAL_DATASETS):
        ds = sub[sub["dataset"] == ds_key]
        positions = np.arange(len(models))
        jitter_w = 0.12

        for i, m in enumerate(models):
            vals = ds[ds["model"] == m]["ndcg@10"].values
            color = MODEL_COLORS[m]
            xs = np.full_like(vals, i) + np.random.default_rng(42).uniform(
                -jitter_w, jitter_w, size=len(vals)
            )
            ax.scatter(
                xs, vals, color=color, s=55, edgecolor="black", linewidth=0.5, zorder=4, alpha=0.85
            )
            mean_val = vals.mean()
            ax.hlines(mean_val, i - 0.25, i + 0.25, color=color, linewidth=2.5, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(models, fontsize=11, rotation=20, ha="right")
        ax.set_title(ds_label, fontsize=13)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("NDCG@10 (tuned)")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"real_hybrid_seeds.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote real_hybrid_seeds.{pdf,png}")


def fig_ablation_heatmap(df):
    all_models = set()
    for parent, abls in ABLATION_GROUPS:
        all_models.add(parent)
        all_models.update(abls)

    synth = df[df["batch"].isin(["ABLATION_BATCH_SYNTH", "DEFAULT_BATCH_SYNTH"])].copy()
    synth["topo"] = synth["dataset"].str.extract(FAMILY_RE)

    synth_means = (
        synth[synth["model"].isin(all_models)]
        .groupby(["topo", "model"])["ndcg@10"]
        .mean()
        .unstack()
    )

    real = df[
        df["batch"].isin(["ABLATION_BATCH_REAL", "DEFAULT_BATCH_REAL"])
        & (df["dataset"] == "douban-book")
    ]
    real_means = real[real["model"].isin(all_models)].groupby("model")["ndcg@10"].mean()

    abl_names = []
    for _, abls in ABLATION_GROUPS:
        abl_names.extend(abls)

    cols = [t for t in TOPO_ORDER if t in synth_means.index] + ["Douban-Book"]
    matrix = np.full((len(abl_names), len(cols)), np.nan)

    for i, abl in enumerate(abl_names):
        parent = [p for p, al in ABLATION_GROUPS if abl in al][0]
        for j, col in enumerate(cols):
            if col == "Douban-Book":
                if abl in real_means.index and parent in real_means.index:
                    matrix[i, j] = real_means[abl] - real_means[parent]
            else:
                if col in synth_means.index:
                    if abl in synth_means.columns and parent in synth_means.columns:
                        matrix[i, j] = synth_means.loc[col, abl] - synth_means.loc[col, parent]

    vmax = np.nanmax(np.abs(matrix))  # noqa: F841
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, aspect="auto")

    col_labels = [TOPO_PRETTY.get(c, c) for c in cols]
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=11, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(abl_names)))
    ax.set_yticklabels([ABL_PRETTY.get(a, a) for a in abl_names], fontsize=11)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            norm_v = (
                (v - matrix[~np.isnan(matrix)].min())
                / (matrix[~np.isnan(matrix)].max() - matrix[~np.isnan(matrix)].min())
                if matrix[~np.isnan(matrix)].max() != matrix[~np.isnan(matrix)].min()
                else 0.5
            )
            text_color = "white" if norm_v < 0.45 else "black"
            ax.text(j, i, f"{v:+.4f}", ha="center", va="center", fontsize=9, color=text_color)

    # separator line before Douban-Book column
    ax.axvline(len(cols) - 1.5, color="black", lw=1.2, ls="--")

    # separator lines between ablation groups
    row_idx = 0
    for _, abls in ABLATION_GROUPS[:-1]:
        row_idx += len(abls)
        ax.axhline(row_idx - 0.5, color="black", lw=0.8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\Delta$ NDCG@10 vs parent", fontsize=11)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"ablation_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote ablation_heatmap.{pdf,png}")


def main():
    df = load()
    fig_hybrid_sweep_gaps(df)
    fig_real_hybrid_seeds(df)
    fig_ablation_heatmap(df)


if __name__ == "__main__":
    main()
