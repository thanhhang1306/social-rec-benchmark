"""Compare ablation variants against their parent models."""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ABLATION_GROUPS = {
    "HAGN": {
        "parent":    "HAGN",
        "ablations": ["HAGNUniform", "HAGNFixed"],
        "labels":    {
            "HAGN":        "HAGN (full)",
            "HAGNUniform": "− Jaccard",
            "HAGNFixed":   "− gate (α=1)",
        },
        "colors":    {
            "HAGN":        "#C45AB3",
            "HAGNUniform": "#E8A5DB",
            "HAGNFixed":   "#9B2E8A",
        },
    },
    "CSGCN": {
        "parent":    "CSGCN",
        "ablations": ["CSGCNNoWarmup"],
        "labels":    {
            "CSGCN":          "CSGCN (full)",
            "CSGCNNoWarmup":  "− warmup",
        },
        "colors":    {
            "CSGCN":          "#8CB369",
            "CSGCNNoWarmup":  "#C8E6A0",
        },
    },
    "ASLGCN": {
        "parent":    "ASLGCN",
        "ablations": ["ASLGCNSymmetric"],
        "labels":    {
            "ASLGCN":          "ASLGCN (full)",
            "ASLGCNSymmetric": "− asymmetry",
        },
        "colors":    {
            "ASLGCN":          "#F4A261",
            "ASLGCNSymmetric": "#FCD5AA",
        },
    },
}

SYNTH_TYPES = [
    "contrarian", "echo_chamber", "fake_users", "line_graph",
    "partial_alignment", "random", "scale_free", "small_world", "star",
]
REAL_DATASETS = ["yelp", "douban-book", "lastfm"]

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


def collect_results(dirs: list) -> pd.DataFrame:
    frames = []
    for d in dirs:
        for root, _, files in os.walk(d):
            if "final_metrics.csv" in files:
                frames.append(pd.read_csv(os.path.join(root, "final_metrics.csv")))
    if not frames:
        raise FileNotFoundError(f"No final_metrics.csv found under: {dirs}")
    return pd.concat(frames, ignore_index=True)


def _classify_topology(name: str) -> str:
    for t in SYNTH_TYPES:
        if name.startswith(t):
            return t
    return name


def load_and_merge(ablation_dirs, baseline_dirs) -> pd.DataFrame:
    abl = collect_results(ablation_dirs)
    base = collect_results(baseline_dirs)

    # keep only parent models from baseline that have ablation counterparts
    parents = {g["parent"] for g in ABLATION_GROUPS.values()}
    base = base[base["Model"].isin(parents)].copy()

    df = pd.concat([abl, base], ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["dataset_type"] = df["dataset"].apply(
        lambda n: "real" if n in REAL_DATASETS else "synthetic"
    )
    df["topology"] = df["dataset"].apply(_classify_topology)

    all_models = {g["parent"] for g in ABLATION_GROUPS.values()} | \
                 {m for g in ABLATION_GROUPS.values() for m in g["ablations"]}
    df = df[df["model"].isin({m.lower() for m in all_models} |
                              {m for m in all_models})]

    # normalize model name capitalization
    model_map = {m.lower(): m for m in all_models}
    df["model"] = df["model"].apply(lambda m: model_map.get(m.lower(), m))

    n_rows = len(df)
    print(f"  Loaded {n_rows:,} rows | {df['model'].nunique()} models | "
          f"{df['dataset'].nunique()} datasets")
    return df


def avg_seeds(df: pd.DataFrame, groupby: list, metric: str) -> pd.DataFrame:
    return df.groupby(groupby)[metric].mean().reset_index()


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_grouped_bars(df: pd.DataFrame, metric: str, out_dir: str):
    """One figure per ablation family; bars grouped by dataset_type."""
    for family, cfg in ABLATION_GROUPS.items():
        models = [cfg["parent"]] + cfg["ablations"]
        labels = cfg["labels"]
        colors = cfg["colors"]

        sub = df[df["model"].isin(models)].copy()
        agg = avg_seeds(sub, ["model", "dataset_type"], metric)

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(models))
        w = 0.35
        dtype_offsets = {"real": -w / 2, "synthetic": w / 2}
        dtype_hatches = {"real": "", "synthetic": "///"}

        for dtype, offset in dtype_offsets.items():
            vals = []
            for m in models:
                row = agg[(agg["model"] == m) & (agg["dataset_type"] == dtype)]
                vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            bars = ax.bar(
                x + offset, vals, w,
                color=[colors[m] for m in models],
                hatch=dtype_hatches[dtype],
                edgecolor="white", linewidth=0.5,
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in models], rotation=15, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{family} Ablations — {metric.upper()}")

        ymax = agg[metric].max()
        ax.set_ylim(0, ymax * 1.25)

        # legend for dataset type — placed outside plot area to avoid overlap
        real_patch  = mpatches.Patch(facecolor="grey", label="Real datasets")
        synth_patch = mpatches.Patch(facecolor="grey", hatch="///",
                                     label="Synthetic datasets")
        ax.legend(handles=[real_patch, synth_patch],
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

        tag = metric.replace("@", "at")
        _save(fig, os.path.join(out_dir, f"ablation_{family.lower()}_bars_{tag}.pdf"))


def plot_delta_heatmap(df: pd.DataFrame, metric: str, out_dir: str):
    """Heatmap of Δ metric (ablation − parent): rows = variants, cols = topologies."""
    rows = []
    for family, cfg in ABLATION_GROUPS.items():
        parent = cfg["parent"]
        for abl in cfg["ablations"]:
            label = cfg["labels"][abl]
            for dtype in ["real", "synthetic"]:
                sub = df[df["dataset_type"] == dtype].copy()
                topologies = REAL_DATASETS if dtype == "real" else SYNTH_TYPES
                for topo in topologies:
                    if dtype == "real":
                        sub_t = sub[sub["dataset"] == topo]
                    else:
                        sub_t = sub[sub["topology"] == topo]
                    if sub_t.empty:
                        continue
                    p_val = sub_t[sub_t["model"] == parent][metric].mean()
                    a_val = sub_t[sub_t["model"] == abl][metric].mean()
                    if np.isnan(p_val) or np.isnan(a_val):
                        continue
                    rows.append({
                        "ablation": label,
                        "topology": topo,
                        "delta": a_val - p_val,
                        "dataset_type": dtype,
                    })

    if not rows:
        print("  No delta data — skipping heatmap.")
        return

    delta_df = pd.DataFrame(rows)
    pivot = delta_df.pivot_table(index="ablation", columns="topology",
                                 values="delta", aggfunc="mean")

    # order columns: real datasets first, then synthetic topologies
    col_order = [c for c in REAL_DATASETS if c in pivot.columns] + \
                [c for c in SYNTH_TYPES if c in pivot.columns]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.9), 3.5))
    vmax = delta_df["delta"].abs().quantile(0.95)
    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".3f", annot_kws={"size": 7},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": f"Δ {metric.upper()} (ablation − parent)"},
    )
    ax.set_title(f"Ablation Impact: Δ {metric.upper()} vs. Parent Model")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    # separator line between real and synthetic columns
    n_real = sum(1 for c in REAL_DATASETS if c in pivot.columns)
    if 0 < n_real < len(pivot.columns):
        ax.axvline(x=n_real, color="black", linewidth=1.5, linestyle="--")
        ax.text(n_real / 2, -0.4, "real", ha="center", va="top",
                fontsize=8, transform=ax.get_xaxis_transform())
        ax.text(n_real + (len(pivot.columns) - n_real) / 2, -0.4, "synthetic",
                ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())

    tag = metric.replace("@", "at")
    _save(fig, os.path.join(out_dir, f"ablation_delta_heatmap_{tag}.pdf"))


def plot_real_dataset_bars(df: pd.DataFrame, metric: str, out_dir: str):
    """Grouped bars per real dataset: one cluster per ablation family."""
    real_df = df[df["dataset_type"] == "real"].copy()
    datasets = [d for d in REAL_DATASETS if d in real_df["dataset"].unique()]
    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(4.5 * len(datasets), 4.5),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = real_df[real_df["dataset"] == dataset]
        tick_labels, vals, colors = [], [], []

        for family, cfg in ABLATION_GROUPS.items():
            models = [cfg["parent"]] + cfg["ablations"]
            for m in models:
                row = sub[sub["model"] == m]
                v = float(row[metric].mean()) if len(row) else np.nan
                tick_labels.append(cfg["labels"][m])
                vals.append(v)
                colors.append(cfg["colors"][m])

        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(dataset)
        if ax == axes[0]:
            ax.set_ylabel(metric.upper())
        ymax = max(v for v in vals if not np.isnan(v)) if any(not np.isnan(v) for v in vals) else 1
        ax.set_ylim(0, ymax * 1.25)

        # separator lines between families
        n_models = [1 + len(cfg["ablations"]) for cfg in ABLATION_GROUPS.values()]
        boundaries = np.cumsum(n_models[:-1]) - 0.5
        for b in boundaries:
            ax.axvline(x=b, color="grey", linewidth=0.8, linestyle=":")

    fig.suptitle(f"Real Dataset Ablations — {metric.upper()}", fontsize=12, y=1.01)
    tag = metric.replace("@", "at")
    _save(fig, os.path.join(out_dir, f"ablation_real_datasets_{tag}.pdf"))


def plot_topology_lines(df: pd.DataFrame, metric: str, out_dir: str):
    """Line plots per topology: parent vs. ablation(s) across parameter variants."""
    synth = df[df["dataset_type"] == "synthetic"].copy()

    for family, cfg in ABLATION_GROUPS.items():
        models = [cfg["parent"]] + cfg["ablations"]
        sub = synth[synth["model"].isin(models)]
        topologies = sorted(sub["topology"].dropna().unique())
        if not topologies:
            continue

        ncols = 3
        nrows = (len(topologies) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()

        for ax, topo in zip(axes, topologies):
            topo_sub = sub[sub["topology"] == topo]
            agg = avg_seeds(topo_sub, ["model", "dataset"], metric)
            datasets_sorted = sorted(agg["dataset"].unique())

            for m in models:
                m_agg = agg[agg["model"] == m].set_index("dataset")
                y = [m_agg.loc[d, metric] if d in m_agg.index else np.nan
                     for d in datasets_sorted]
                ax.plot(range(len(datasets_sorted)), y,
                        label=cfg["labels"][m],
                        color=cfg["colors"][m],
                        marker="o", markersize=3, linewidth=1.5)

            ax.set_title(topo, fontsize=9)
            ax.set_xticks([])
            ax.set_ylabel(metric.upper() if ax in axes[::ncols] else "")
            ax.legend(fontsize=7, loc="best")

        for ax in axes[len(topologies):]:
            ax.set_visible(False)

        fig.suptitle(f"{family} Ablations by Topology — {metric.upper()}",
                     fontsize=12, y=1.01)
        tag = metric.replace("@", "at")
        _save(fig, os.path.join(out_dir,
              f"ablation_{family.lower()}_topology_lines_{tag}.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_dirs", nargs="+",
                        default=["results/ABLATION_BATCH_REAL",
                                 "results/ABLATION_BATCH_SYNTH"])
    parser.add_argument("--baseline_dirs", nargs="+",
                        default=["results/DEFAULT_BATCH_REAL",
                                 "results/DEFAULT_BATCH_SYNTH"])
    parser.add_argument("--metrics", nargs="+",
                        default=["ndcg@10", "recall@10", "mrr@10", "precision@10", "hit@10"])
    parser.add_argument("--out_dir", default="exploration/ablations")
    args = parser.parse_args()

    print("Loading data...")
    df = load_and_merge(args.ablation_dirs, args.baseline_dirs)

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"  Skipping '{metric}' — not in columns.")
            continue
        print(f"\nPlotting: {metric}")
        plot_grouped_bars(df, metric, args.out_dir)
        plot_delta_heatmap(df, metric, args.out_dir)
        plot_real_dataset_bars(df, metric, args.out_dir)
        plot_topology_lines(df, metric, args.out_dir)
    print(f"\nDone! Figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
