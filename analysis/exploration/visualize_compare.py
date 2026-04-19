"""Cross-dataset comparison and multi-seed reproducibility plots."""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

SYNTH_TYPES = [
    "contrarian", "echo_chamber", "fake_users", "line_graph",
    "partial_alignment", "random", "scale_free", "small_world", "star",
]

# one distinct color per topology (colorblind-friendly)
TOPO_PALETTE = {
    "contrarian":       "#E24B4A",
    "echo_chamber":     "#EF9F27",
    "fake_users":       "#F4A261",
    "line_graph":       "#8CB369",
    "partial_alignment":"#1D9E75",
    "random":           "#2D6A9F",
    "scale_free":       "#7F77DD",
    "small_world":      "#C45AB3",
    "star":             "#9BA3AF",
    "real":             "#1C2331",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def topo_legend(ax):
    patches = [mpatches.Patch(color=TOPO_PALETTE.get(t, "#888"), label=t.replace("_", " "))
               for t in SYNTH_TYPES + ["real"]]
    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0.5),
              framealpha=0.5, edgecolor="none", fontsize=8)


def load_stats(csv_paths: list) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"  Warning: {p} not found, skipping.")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No dataset_stats.csv files found.")
    df = pd.concat(frames, ignore_index=True)
    if "topology" not in df.columns:
        df["topology"] = df["dataset"].apply(
            lambda n: next((t for t in SYNTH_TYPES if n.startswith(t)), "real")
        )
    print(f"  Loaded {len(df):,} rows | {df['dataset'].nunique()} unique datasets "
          f"| {df['dseed'].nunique()} data seed(s)")
    return df


def _mean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Average numeric columns over data seeds, keeping one row per dataset."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    meta    = ["dataset", "topology"]
    return df.groupby(meta)[numeric].mean().reset_index()


def fig_scatter_overview(df: pd.DataFrame, out: str):
    """2×2 scatter of key stat pairs, colored by topology."""
    mean = _mean_df(df)
    synth = mean[mean["topology"] != "real"]

    pairs = [
        ("param_sparsity",    "density",           "Sparsity (param)",    "Actual density"),
        ("param_noise",       "gini_user",          "Noise ratio (param)", "User interaction Gini"),
        ("param_n_items",     "catalog_coverage",   "N items (param)",     "Catalog coverage"),
        ("avg_social_degree", "density",            "Avg social degree",   "Actual density"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, pairs):
        sub = synth.dropna(subset=[xcol, ycol])
        if sub.empty:
            ax.set_visible(False)
            continue
        for topo in SYNTH_TYPES:
            t = sub[sub["topology"] == topo]
            if t.empty:
                continue
            ax.scatter(t[xcol], t[ycol], color=TOPO_PALETTE[topo],
                       alpha=0.65, s=30, label=topo.replace("_", " "), zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # shared legend on last axis
    topo_legend(axes[1, 1])
    fig.suptitle("Dataset overview — key stat pairs (colored by topology)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out, "scatter_overview.png"))


def fig_stats_by_topology(df: pd.DataFrame, out: str):
    """4-panel box plots showing distribution of key stats across topologies."""
    stats_to_plot = [
        ("density",           "Interaction density"),
        ("gini_user",         "User interaction Gini"),
        ("gini_item",         "Item interaction Gini"),
        ("catalog_coverage",  "Catalog coverage"),
    ]

    synth = df[df["topology"] != "real"].copy()
    synth["topology_label"] = synth["topology"].str.replace("_", "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, label) in zip(axes.flat, stats_to_plot):
        sub = synth.dropna(subset=[col])
        if sub.empty:
            ax.set_visible(False)
            continue
        order = [t for t in SYNTH_TYPES if t in sub["topology"].unique()]
        palette = {t: TOPO_PALETTE[t] for t in order}
        sns.boxplot(data=sub, x="topology", y=col, order=order, palette=palette,
                    ax=ax, linewidth=0.8, fliersize=3)
        # overlay individual seed points if multi-seed
        if df["dseed"].nunique() > 1:
            sns.stripplot(data=sub, x="topology", y=col, order=order,
                          color="black", alpha=0.3, size=3, ax=ax, jitter=True)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticklabels([t.replace("_", "\n") for t in order], fontsize=8)

    fig.suptitle("Dataset stats by topology", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out, "stats_by_topology.png"))


def fig_ablation_lines(df: pd.DataFrame, out: str):
    """Line plots of density and gini_user vs sparsity/noise, one line per topology."""
    mean = _mean_df(df)
    synth = df[df["topology"] != "real"].copy()
    n_seeds = df["dseed"].nunique()

    ablations = [
        ("param_sparsity", "density",   "Sparsity (param)", "Interaction density"),
        ("param_sparsity", "gini_user", "Sparsity (param)", "User Gini"),
        ("param_noise",    "density",   "Noise ratio",      "Interaction density"),
        ("param_noise",    "gini_user", "Noise ratio",      "User Gini"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, ablations):
        for topo in SYNTH_TYPES:
            sub = synth[synth["topology"] == topo].dropna(subset=[xcol, ycol])
            if sub.empty:
                continue
            grp = sub.groupby(xcol)[ycol]
            xvals = sorted(grp.mean().index)
            ymean = grp.mean().reindex(xvals).values
            ystd  = grp.std().reindex(xvals).fillna(0).values

            color = TOPO_PALETTE[topo]
            ax.plot(xvals, ymean, color=color, linewidth=1.5, marker="o", markersize=4,
                    label=topo.replace("_", " "))
            if n_seeds > 1:
                ax.fill_between(xvals, ymean - ystd, ymean + ystd,
                                color=color, alpha=0.15)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # legend on last panel only
    topo_legend(axes[1, 1])
    seed_note = f" (band = ±1 std, {n_seeds} seeds)" if n_seeds > 1 else ""
    fig.suptitle(f"Ablation lines — sparsity & noise effect on density/Gini{seed_note}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out, "ablation_lines.png"))


def fig_seed_reproducibility(df: pd.DataFrame, out: str):
    """Bar chart of coefficient of variation across seeds (multi-seed only)."""
    if df["dseed"].nunique() < 2:
        print("  Skipping reproducibility figure — only 1 data seed present.")
        return

    stat_cols = ["density", "gini_user", "gini_item",
                 "avg_social_degree", "clustering", "catalog_coverage"]
    stat_labels = ["Density", "Gini user", "Gini item",
                   "Avg degree", "Clustering", "Coverage"]

    synth = df[df["topology"] != "real"]
    records = []
    for topo in SYNTH_TYPES:
        sub = synth[synth["topology"] == topo]
        for col, label in zip(stat_cols, stat_labels):
            vals = sub.groupby("dataset")[col].std() / (sub.groupby("dataset")[col].mean().abs() + 1e-12)
            records.append({
                "topology": topo,
                "stat":     label,
                "cv_mean":  vals.mean(),
                "cv_std":   vals.std(),
            })
    cv_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(stat_cols))
    w = 0.8 / len(SYNTH_TYPES)
    for i, topo in enumerate(SYNTH_TYPES):
        sub = cv_df[cv_df["topology"] == topo]
        vals  = [sub.loc[sub["stat"] == lbl, "cv_mean"].values[0]
                 if len(sub.loc[sub["stat"] == lbl]) > 0 else 0
                 for lbl in stat_labels]
        errs  = [sub.loc[sub["stat"] == lbl, "cv_std"].values[0]
                 if len(sub.loc[sub["stat"] == lbl]) > 0 else 0
                 for lbl in stat_labels]
        offset = (i - len(SYNTH_TYPES) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w * 0.9, yerr=errs, capsize=2,
               color=TOPO_PALETTE[topo], alpha=0.85, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(stat_labels)
    ax.set_ylabel("Coefficient of variation (std / mean) across seeds")
    ax.set_title(f"Dataset reproducibility across {df['dseed'].nunique()} data seeds "
                 f"(lower = more reproducible)")
    patches = [mpatches.Patch(color=TOPO_PALETTE[t], label=t.replace("_", " "))
               for t in SYNTH_TYPES]
    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0.5),
              framealpha=0.5, edgecolor="none", fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(out, "seed_reproducibility.png"))


def fig_topology_comparison_errorbars(df: pd.DataFrame, out: str):
    """Bar chart of key stats per topology; error bars from multi-seed spread."""
    stat_cols   = ["density", "gini_user", "avg_social_degree", "catalog_coverage"]
    stat_labels = ["Interaction density", "User Gini", "Avg social degree", "Catalog coverage"]

    synth = df[df["topology"] != "real"]
    n_seeds = df["dseed"].nunique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, label) in zip(axes.flat, zip(stat_cols, stat_labels)):
        sub = synth.dropna(subset=[col])
        if sub.empty:
            ax.set_visible(False)
            continue
        grp   = sub.groupby("topology")[col]
        means = grp.mean().reindex(SYNTH_TYPES).fillna(0)
        stds  = grp.std().reindex(SYNTH_TYPES).fillna(0) if n_seeds > 1 else None
        colors = [TOPO_PALETTE[t] for t in SYNTH_TYPES]
        bars = ax.bar(range(len(SYNTH_TYPES)), means.values,
                      yerr=stds.values if stds is not None else None,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.4,
                      capsize=4 if stds is not None else 0)
        ax.set_xticks(range(len(SYNTH_TYPES)))
        ax.set_xticklabels([t.replace("_", "\n") for t in SYNTH_TYPES], fontsize=8)
        ax.set_ylabel(label)
        if stds is not None:
            ax.set_title(f"{label} (mean ± std, {n_seeds} seeds)")
        else:
            ax.set_title(label)

    fig.suptitle("Key stats per topology", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out, "topology_comparison.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset comparison and multi-seed reproducibility")
    parser.add_argument("--stats", type=str, nargs="+", required=True,
                        help="One or more dataset_stats.csv files (one per data seed)")
    parser.add_argument("--out",   type=str, default="exploration/dataset_comparison",
                        help="Output directory")
    args = parser.parse_args()

    print("Loading stats CSVs...")
    df = load_stats(args.stats)

    out = args.out
    n_seeds = df["dseed"].nunique()
    print(f"\nGenerating figures ({n_seeds} data seed(s))...")

    fig_scatter_overview(df, out)
    fig_stats_by_topology(df, out)
    fig_ablation_lines(df, out)
    fig_topology_comparison_errorbars(df, out)
    fig_seed_reproducibility(df, out)

    print(f"\nDone! Figures saved to: {out}/")


if __name__ == "__main__":
    main()
