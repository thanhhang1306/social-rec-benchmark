"""Aggregate analysis of RecBole experiment results."""

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

MODELS = ["BPR", "LightGCN", "SimGCL", "DiffNet", "MHCN", "HAGN", "CSGCN", "ASLGCN"]

PALETTE = {
    "BPR":      "#7F77DD",
    "LightGCN": "#EF9F27",
    "SimGCL":   "#1D9E75",
    "DiffNet":  "#378ADD",
    "MHCN":     "#E24B4A",
    "HAGN":     "#C45AB3",
    "CSGCN":    "#8CB369",
    "ASLGCN":   "#F4A261",
}

MARKERS = {
    "BPR": "o", "LightGCN": "s", "SimGCL": "^", "DiffNet": "P",
    "MHCN": "D", "HAGN": "X", "CSGCN": "v", "ASLGCN": "*",
}

REAL_DATASETS = ["yelp", "douban-book", "lastfm"]

SYNTH_TYPES = [
    "contrarian", "echo_chamber", "fake_users", "line_graph",
    "partial_alignment", "random", "scale_free", "small_world", "star",
]

TEST_METRICS = {
    "recall@10":    "Recall@10",
    "ndcg@10":      "NDCG@10",
    "mrr@10":       "MRR@10",
    "precision@10": "Precision@10",
    "hit@10":       "Hit@10",
}

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


def collect_results(results_dirs: list) -> pd.DataFrame:
    """Walk directories and concatenate all final_metrics.csv files."""
    frames = []
    for d in results_dirs:
        for root, dirs, files in os.walk(d):
            if "final_metrics.csv" in files:
                frames.append(pd.read_csv(os.path.join(root, "final_metrics.csv")))
    if not frames:
        raise FileNotFoundError(f"No final_metrics.csv found under: {results_dirs}")
    return pd.concat(frames, ignore_index=True)


def _classify_topology(name: str) -> str:
    for t in SYNTH_TYPES:
        if name.startswith(t):
            return t
    return name


def load_data(csv_path=None, results_dirs=None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
    elif results_dirs:
        df = collect_results(results_dirs)
    else:
        raise ValueError("Provide either --csv or --results_dir")

    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # tuned bool from Mode column
    df["tuned"] = df["mode"].str.strip().str.lower() == "tuned"

    # normalize dataset_type: 'synth' → 'synthetic'
    if "dataset_type" in df.columns:
        df["dataset_type"] = df["dataset_type"].str.lower().str.replace("^synth$", "synthetic", regex=True)
    else:
        df["dataset_type"] = df["dataset"].apply(
            lambda n: "real" if n in REAL_DATASETS else "synthetic"
        )

    df["topology"] = df["dataset"].apply(_classify_topology)

    n_seeds = df.get("seed", pd.Series(dtype=float)).nunique()
    n_dseeds = df.get("data_seed", pd.Series(dtype=float)).nunique()
    print(f"  {len(df):,} rows | {df['dataset'].nunique()} datasets | "
          f"{df['model'].nunique()} models | "
          f"model seeds: {n_seeds} | data seeds: {n_dseeds}")
    return df


def avg_seeds(df: pd.DataFrame, groupby: list, metrics: list = None) -> pd.DataFrame:
    """Average metric values over seed / data_seed within each group."""
    if metrics is None:
        metrics = list(TEST_METRICS.keys())
    available = [m for m in metrics if m in df.columns]
    return df.groupby(groupby)[available].mean().reset_index()


def legend_patches(models=None):
    if models is None:
        models = MODELS
    return [mpatches.Patch(color=PALETTE[m], label=m) for m in models if m in PALETTE]


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _metric_tag(m: str) -> str:
    return m.replace("@", "at")


def fig_real_datasets(df: pd.DataFrame, out: str, metric: str, tuned: bool):
    sub = df[(df["dataset"].isin(REAL_DATASETS)) & (df["tuned"] == tuned)]
    sub = avg_seeds(sub, ["dataset", "model"], [metric])

    ds_order = ["yelp", "douban-book", "lastfm"]
    n_models = len(MODELS)
    width = 0.10
    x = np.arange(len(ds_order))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(MODELS):
        vals = [
            sub.loc[(sub["dataset"] == ds) & (sub["model"] == model), metric].values[0]
            if len(sub.loc[(sub["dataset"] == ds) & (sub["model"] == model)]) > 0 else 0
            for ds in ds_order
        ]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               color=PALETTE.get(model, "#888"), edgecolor="white", linewidth=0.4, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(["Yelp", "Douban-Book", "LastFM"])
    ax.set_ylabel(TEST_METRICS[metric])
    suffix = "tuned" if tuned else "default"
    ax.set_title(f"{TEST_METRICS[metric]} on real datasets ({suffix})")
    ax.legend(handles=legend_patches(), loc="center left",
              bbox_to_anchor=(1, 0.5), framealpha=0.5, edgecolor="none")
    fig.tight_layout()
    _save(fig, os.path.join(out, "real", f"real_{_metric_tag(metric)}_{suffix}.png"))


def fig_tuning_impact(df: pd.DataFrame, out: str, metric: str):
    rows = []
    for ds in REAL_DATASETS:
        for model in MODELS:
            d = df[(df["dataset"] == ds) & (df["model"] == model)]
            v_def  = avg_seeds(d[~d["tuned"]], ["dataset", "model"], [metric])[metric].values
            v_tune = avg_seeds(d[ d["tuned"]], ["dataset", "model"], [metric])[metric].values
            if len(v_def) and len(v_tune):
                pct = 100 * (v_tune[0] - v_def[0]) / (abs(v_def[0]) + 1e-12)
                rows.append({"dataset": ds, "model": model, "pct_gain": pct})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_df, x="dataset", y="pct_gain", hue="model",
                palette=PALETTE, order=REAL_DATASETS, hue_order=MODELS,
                ax=ax, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticklabels(["Yelp", "Douban-Book", "LastFM"])
    ax.set_ylabel(f"% change in {TEST_METRICS[metric]}")
    ax.set_title(f"Impact of hyperparameter tuning ({TEST_METRICS[metric]})")
    ax.legend(handles=legend_patches(), loc="center left",
              bbox_to_anchor=(1, 0.5), framealpha=0.5, edgecolor="none")
    fig.tight_layout()
    _save(fig, os.path.join(out, "real", f"tuning_impact_{_metric_tag(metric)}.png"))


def fig_metrics_table(df: pd.DataFrame, out: str, dataset: str, tuned: bool):
    sub = avg_seeds(
        df[(df["dataset"] == dataset) & (df["tuned"] == tuned)],
        ["model"], list(TEST_METRICS.keys())
    ).set_index("model").reindex(MODELS).fillna(0)

    if (sub == 0).all().all():
        return

    normed = (sub - sub.min()) / (sub.max() - sub.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(normed.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(TEST_METRICS)))
    ax.set_xticklabels(list(TEST_METRICS.values()), rotation=25, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)

    for i, model in enumerate(MODELS):
        if model not in sub.index:
            continue
        for j, met in enumerate(TEST_METRICS.keys()):
            if met not in sub.columns:
                continue
            val = sub.loc[model, met]
            brightness = normed.loc[model, met]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=8, color="white" if brightness > 0.6 else "black")

    suffix = "tuned" if tuned else "default"
    ax.set_title(f"All metrics — {dataset} ({suffix})")
    fig.colorbar(im, ax=ax, label="Normalized score", shrink=0.8)
    fig.tight_layout()
    safe = dataset.replace("/", "-")
    _save(fig, os.path.join(out, "real", f"metrics_table_{safe}_{suffix}.png"))


def fig_synth_heatmap(df: pd.DataFrame, out: str, metric: str, tuned: bool):
    sub = df[(df["dataset_type"] == "synthetic") & (df["tuned"] == tuned)]
    sub = avg_seeds(sub, ["topology", "model"], [metric])
    pivot = sub.pivot(index="topology", columns="model", values=metric)
    pivot = pivot.reindex(index=SYNTH_TYPES, columns=MODELS).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot * 100, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": f"{TEST_METRICS[metric]} × 100"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([t.replace("_", " ").title() for t in SYNTH_TYPES], rotation=0)
    suffix = "tuned" if tuned else "default"
    ax.set_title(f"{TEST_METRICS[metric]} by topology × model ({suffix}, averaged over params & seeds)")
    fig.tight_layout()
    _save(fig, os.path.join(out, "synth", f"heatmap_{_metric_tag(metric)}_{suffix}.png"))


def fig_real_vs_synth(df: pd.DataFrame, out: str, metric: str, tuned: bool):
    sub = df[df["tuned"] == tuned]
    real_avg  = avg_seeds(sub[sub["dataset_type"] == "real"],
                          ["model"], [metric]).set_index("model")[metric]
    synth_avg = avg_seeds(sub[sub["dataset_type"] == "synthetic"],
                          ["model"], [metric]).set_index("model")[metric]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for model in MODELS:
        if model in real_avg.index and model in synth_avg.index:
            ax.scatter(synth_avg[model], real_avg[model],
                       color=PALETTE.get(model, "#888"), marker=MARKERS.get(model, "o"),
                       s=100, zorder=3)
            ax.annotate(model, (synth_avg[model], real_avg[model]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=PALETTE.get(model, "#888"))

    lo = min(synth_avg.min(), real_avg.min()) * 0.8
    hi = max(synth_avg.max(), real_avg.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.4, label="y = x")
    ax.set_xlabel(f"Avg {TEST_METRICS[metric]} on synthetic")
    ax.set_ylabel(f"Avg {TEST_METRICS[metric]} on real")
    suffix = "tuned" if tuned else "default"
    ax.set_title(f"Real vs synthetic performance ({suffix})")
    ax.legend(loc="lower right", framealpha=0.5, edgecolor="none")
    fig.tight_layout()
    _save(fig, os.path.join(out, "synth", f"real_vs_synth_{_metric_tag(metric)}_{suffix}.png"))


def fig_model_ranks(df: pd.DataFrame, out: str, tuned: bool):
    sub = df[df["tuned"] == tuned]
    metric_cols = [m for m in TEST_METRICS if m in df.columns]
    records = []
    for dtype in ["real", "synthetic"]:
        msub = avg_seeds(sub[sub["dataset_type"] == dtype], ["dataset", "model"], metric_cols)
        for met in metric_cols:
            grp = msub.groupby("dataset")[met].rank(ascending=False)
            tmp = msub.copy()
            tmp["rank"] = grp.values
            avg_rank = tmp.groupby("model")["rank"].mean().reset_index()
            avg_rank["dtype"] = dtype
            records.append(avg_rank)
    if not records:
        return
    rank_df   = pd.concat(records)
    mean_rank = rank_df.groupby(["model", "dtype"])["rank"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(MODELS))
    w = 0.35
    for j, (dtype, hatch) in enumerate([("real", ""), ("synthetic", "//")]):
        vals = [
            mean_rank.loc[(mean_rank["model"] == m) & (mean_rank["dtype"] == dtype), "rank"].values[0]
            if len(mean_rank.loc[(mean_rank["model"] == m) & (mean_rank["dtype"] == dtype)]) > 0
            else np.nan
            for m in MODELS
        ]
        offset = (j - 0.5) * w
        ax.bar(x + offset, vals, w * 0.92,
               color=[PALETTE.get(m, "#888") for m in MODELS],
               alpha=0.9 if dtype == "real" else 0.5,
               hatch=hatch, edgecolor="white", linewidth=0.4,
               label=dtype.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("Average rank (lower = better)")
    suffix = "tuned" if tuned else "default"
    ax.set_title(f"Average model rank across all metrics ({suffix})")
    ax.invert_yaxis()
    extra = [
        mpatches.Patch(facecolor="gray", alpha=0.9, label="Solid = real"),
        mpatches.Patch(facecolor="gray", alpha=0.5, hatch="//", label="Hatched = synthetic"),
    ]
    ax.legend(handles=legend_patches() + extra,
              loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.5, edgecolor="none")
    fig.tight_layout()
    _save(fig, os.path.join(out, "synth", f"model_ranks_{suffix}.png"))


def _line_plot(sub: pd.DataFrame, x_col: str, metric: str, x_label: str,
               title: str, out_path: str):
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODELS:
        d = sub[sub["model"] == model].sort_values(x_col)
        if d.empty:
            continue
        ax.plot(d[x_col], d[metric],
                marker=MARKERS.get(model, "o"), color=PALETTE.get(model, "#888"),
                linewidth=1.5, markersize=6, label=model)
    ax.set_xlabel(x_label)
    ax.set_ylabel(TEST_METRICS.get(metric, metric))
    ax.set_title(title)
    ax.legend(handles=legend_patches(), loc="center left",
              bbox_to_anchor=(1, 0.5), framealpha=0.5, edgecolor="none")
    fig.tight_layout()
    _save(fig, out_path)


def _ablation(df: pd.DataFrame, out: str, metric: str, tuned: bool,
              filter_fn, x_col: str, x_label: str, title_tpl: str, fname: str):
    sub = filter_fn(df[df["tuned"] == tuned].copy())
    sub = avg_seeds(sub, [x_col, "model"], [metric])
    suffix = "tuned" if tuned else "default"
    _line_plot(
        sub, x_col, metric, x_label,
        title=title_tpl.format(metric=TEST_METRICS[metric], suffix=suffix),
        out_path=os.path.join(out, "ablations", f"{fname}_{_metric_tag(metric)}_{suffix}.png"),
    )


def fig_sparsity_ablation(df, out, metric, topology, tuned):
    def f(sub):
        sub = sub[sub["topology"] == topology].copy()
        sub["sp"] = sub["dataset"].str.extract(r"sp([0-9.]+)").astype(float)
        return sub.dropna(subset=["sp"])
    _ablation(df, out, metric, tuned, f, "sp", "Interaction sparsity",
              f"{{metric}} vs sparsity ({topology}, {{suffix}})",
              f"sparsity_{topology}")


def fig_noise_ablation(df, out, metric, topology, tuned):
    def f(sub):
        sub = sub[sub["topology"] == topology].copy()
        sub["ns"] = sub["dataset"].str.extract(r"ns([0-9.]+)").astype(float)
        return sub.dropna(subset=["ns"])
    _ablation(df, out, metric, tuned, f, "ns", "Noise ratio",
              f"{{metric}} vs noise ratio ({topology}, {{suffix}})",
              f"noise_{topology}")


def fig_fake_ratio_ablation(df, out, metric, tuned):
    def f(sub):
        sub = sub[sub["topology"] == "fake_users"].copy()
        sub["fake"] = sub["dataset"].str.extract(r"fake([0-9.]+)").astype(float)
        return sub.dropna(subset=["fake"])
    _ablation(df, out, metric, tuned, f, "fake", "Fake user ratio",
              "{metric} vs fake user ratio ({suffix})", "fake_ratio")


def fig_shared_dims_ablation(df, out, metric, tuned):
    def f(sub):
        sub = sub[sub["topology"] == "partial_alignment"].copy()
        sub["sd"] = sub["dataset"].str.extract(r"_sd(\d+)").astype(float)
        return sub.dropna(subset=["sd"])
    _ablation(df, out, metric, tuned, f, "sd", "Shared latent dimensions",
              "{metric} vs shared dims — partial alignment ({suffix})", "shared_dims")


def fig_avg_degree_ablation(df, out, metric, tuned):
    def f(sub):
        sub["deg"] = sub["dataset"].str.extract(r"_deg([0-9.]+)").astype(float)
        return sub.dropna(subset=["deg"])
    _ablation(df, out, metric, tuned, f, "deg", "Avg social degree",
              "{metric} vs average social degree ({suffix})", "avg_degree")


def fig_user_scale_ablation(df, out, metric, tuned):
    def f(sub):
        sub["n_users"] = sub["dataset"].str.extract(r"_u(\d+)_").astype(float)
        return sub.dropna(subset=["n_users"])
    _ablation(df, out, metric, tuned, f, "n_users", "Number of users",
              "{metric} vs user scale ({suffix})", "user_scale")


def fig_catalog_size_ablation(df, out, metric, tuned):
    def f(sub):
        sub["n_items"] = sub["dataset"].str.extract(r"_i(\d+)_").astype(float)
        return sub.dropna(subset=["n_items"])
    _ablation(df, out, metric, tuned, f, "n_items", "Catalog size (items)",
              "{metric} vs catalog size ({suffix})", "catalog_size")


def main():
    parser = argparse.ArgumentParser(description="Aggregate analysis of RecBole experiment results")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv",         type=str,
                     help="Path to a pre-built combined CSV")
    src.add_argument("--results_dir", type=str, nargs="+",
                     help="One or more results directories to scan for final_metrics.csv")
    parser.add_argument("--out",    type=str, default="exploration/analysis",
                        help="Output root directory")
    parser.add_argument("--metric", type=str, default="recall@10",
                        choices=list(TEST_METRICS.keys()),
                        help="Primary metric for most figures")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(csv_path=args.csv, results_dirs=args.results_dir)

    out    = args.out
    metric = args.metric

    print("\nGenerating real-dataset figures...")
    for tuned in [True, False]:
        fig_real_datasets(df, out, metric, tuned)
        for ds in REAL_DATASETS:
            fig_metrics_table(df, out, ds, tuned)
    fig_tuning_impact(df, out, metric)

    print("\nGenerating synthetic overview figures...")
    for tuned in [True, False]:
        fig_synth_heatmap(df, out, metric, tuned)
        fig_real_vs_synth(df, out, metric, tuned)
        fig_model_ranks(df, out, tuned)

    print("\nGenerating ablation figures...")
    for tuned in [True, False]:
        for topo in ["random", "echo_chamber", "contrarian", "partial_alignment"]:
            fig_sparsity_ablation(df, out, metric, topo, tuned)
            fig_noise_ablation(df, out, metric, topo, tuned)
        fig_fake_ratio_ablation(df, out, metric, tuned)
        fig_shared_dims_ablation(df, out, metric, tuned)
        fig_avg_degree_ablation(df, out, metric, tuned)
        fig_user_scale_ablation(df, out, metric, tuned)
        fig_catalog_size_ablation(df, out, metric, tuned)

    print(f"\nDone! Figures saved to: {out}/")


if __name__ == "__main__":
    main()
