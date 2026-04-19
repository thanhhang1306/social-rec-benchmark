"""Per-dataset interaction and social graph visualizations."""

import argparse
import os
import re
import sys
import warnings
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

warnings.filterwarnings("ignore")


class Tee:
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


PALETTE = {
    "primary":    "#2D6A9F",
    "secondary":  "#E07B39",
    "accent":     "#4CAF82",
    "muted":      "#9BA3AF",
    "background": "#F8F9FA",
    "dark":       "#1C2331",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["background"],
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DADDE1",
    "axes.labelcolor":   PALETTE["dark"],
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "grid.color":        "#EAECEF",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "text.color":        PALETTE["dark"],
})


def load_inter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    print(f"    Loaded .inter: {len(df):,} rows  |  columns: {list(df.columns)}")
    return df

def load_net(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    df.columns = ["source", "target"]
    print(f"    Loaded .net:   {len(df):,} edges")
    return df

def build_nx_graph(net_df: pd.DataFrame) -> nx.DiGraph:
    return nx.from_pandas_edgelist(
        net_df, source="source", target="target", create_using=nx.DiGraph()
    )

def parse_dataset_params(name: str) -> dict:
    """Extract numeric parameters encoded in a dataset name."""
    params = {}
    for key, pat in [
        ("n_users",  r"_u(\d+)_"),
        ("n_items",  r"_i(\d+)_"),
        ("sparsity", r"_sp([0-9.]+)_"),
        ("noise",    r"_ns([0-9.]+)"),
        ("fake",     r"_fake([0-9.]+)"),
        ("sd",       r"_sd(\d+)"),
        ("deg",      r"_deg([0-9.]+)"),
    ]:
        m = re.search(pat, name)
        if m:
            params[key] = float(m.group(1))
    return params


def plot_interaction_overview(inter_df: pd.DataFrame, ax_list: list, has_rating: bool):
    user_counts = inter_df["user_id"].value_counts()
    item_counts = inter_df["item_id"].value_counts()

    ax = ax_list[0]
    ax.hist(user_counts.values, bins=50, color=PALETTE["primary"], alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.set_title("User Interaction Count Distribution")
    ax.set_xlabel("Interactions per User")
    ax.set_ylabel("Number of Users")
    ax.axvline(user_counts.mean(), color=PALETTE["secondary"], linestyle="--", linewidth=1.5,
               label=f"Mean = {user_counts.mean():.1f}")
    ax.axvline(user_counts.median(), color=PALETTE["accent"], linestyle=":", linewidth=1.5,
               label=f"Median = {user_counts.median():.0f}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    ax = ax_list[1]
    ax.hist(item_counts.values, bins=50, color=PALETTE["secondary"], alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.set_title("Item Interaction Count Distribution")
    ax.set_xlabel("Interactions per Item")
    ax.set_ylabel("Number of Items")
    ax.axvline(item_counts.mean(), color=PALETTE["primary"], linestyle="--", linewidth=1.5,
               label=f"Mean = {item_counts.mean():.1f}")
    ax.axvline(item_counts.median(), color=PALETTE["accent"], linestyle=":", linewidth=1.5,
               label=f"Median = {item_counts.median():.0f}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    if has_rating and len(ax_list) > 2:
        ax = ax_list[2]
        ratings = inter_df["rating"].dropna()
        rating_counts = ratings.value_counts().sort_index()
        bars = ax.bar(rating_counts.index, rating_counts.values,
                      color=PALETTE["accent"], alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_title("Rating Distribution")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_xticks(sorted(ratings.unique()))
        for bar, val in zip(bars, rating_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(rating_counts) * 0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=7, color=PALETTE["muted"])
        ax.grid(axis="y", alpha=0.4)


def plot_popularity_analysis(inter_df: pd.DataFrame, dataset_name: str,
                             ax_ccdf: plt.Axes, ax_coverage: plt.Axes):
    """Item popularity CCDF (log-log) and catalog coverage."""
    item_counts = inter_df["item_id"].value_counts().values
    sorted_counts = np.sort(item_counts)[::-1]
    ccdf = 1 - np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    ax_ccdf.loglog(sorted_counts, ccdf, color=PALETTE["primary"], alpha=0.8, linewidth=1.5)
    ax_ccdf.set_title("Item Popularity CCDF (log-log)")
    ax_ccdf.set_xlabel("Interactions per Item")
    ax_ccdf.set_ylabel("P(X ≥ x)")
    ax_ccdf.grid(alpha=0.3)

    params = parse_dataset_params(dataset_name)
    n_items_total = int(params.get("n_items", 0))
    n_items_seen  = inter_df["item_id"].nunique()

    if n_items_total > 0:
        n_unseen = n_items_total - n_items_seen
        coverage = n_items_seen / n_items_total
        bars = ax_coverage.bar(
            ["In interactions", "Never interacted"],
            [n_items_seen, n_unseen],
            color=[PALETTE["accent"], PALETTE["muted"]], alpha=0.85,
            edgecolor="white", linewidth=0.4,
        )
        for bar, val in zip(bars, [n_items_seen, n_unseen]):
            ax_coverage.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() * 0.5, f"{val:,}",
                             ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax_coverage.set_title(f"Catalog Coverage: {coverage:.1%}")
        ax_coverage.set_ylabel("Number of Items")
    else:
        ax_coverage.text(0.5, 0.5, f"{n_items_seen:,} unique items\n(total catalog unknown)",
                         transform=ax_coverage.transAxes, ha="center", va="center", fontsize=10)
        ax_coverage.set_title("Item Coverage")
    ax_coverage.grid(axis="y", alpha=0.4)


def plot_degree_distribution(G: nx.DiGraph, ax_in: plt.Axes, ax_out: plt.Axes):
    in_degrees  = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    for ax, degrees, label, color in [
        (ax_in,  in_degrees,  "In-Degree",  PALETTE["primary"]),
        (ax_out, out_degrees, "Out-Degree", PALETTE["secondary"]),
    ]:
        if max(degrees) == 0:
            ax.text(0.5, 0.5, "No edges", transform=ax.transAxes, ha="center")
            continue
        counts = Counter(degrees)
        x = sorted(counts.keys())
        y = [counts[k] for k in x]
        ax.scatter(x, y, color=color, alpha=0.7, s=18, zorder=3, label="Empirical")

        x_pos = [xi for xi, yi in zip(x, y) if xi > 0 and yi > 0]
        y_pos = [yi for xi, yi in zip(x, y) if xi > 0 and yi > 0]
        if len(x_pos) > 3:
            log_x = np.log(x_pos)
            log_y = np.log(y_pos)
            coeffs = np.polyfit(log_x, log_y, 1)
            x_fit = np.linspace(min(x_pos), max(x_pos), 100)
            y_fit = np.exp(coeffs[1]) * x_fit ** coeffs[0]
            ax.plot(x_fit, y_fit, color=PALETTE["muted"], linestyle="--",
                    linewidth=1, label=f"Power law fit γ={-coeffs[0]:.2f}", zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Social Graph {label} Distribution (log-log)")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)


def plot_graph_sample(G: nx.DiGraph, ax: plt.Axes, n_sample: int, title: str):
    if G.number_of_nodes() <= n_sample:
        G_sub = G
    else:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:n_sample]
        G_sub = G.subgraph(top_nodes).copy()

    degrees = dict(G_sub.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [10 + 200 * (degrees[n] / max_deg) for n in G_sub.nodes()]

    in_degrees = dict(G_sub.in_degree())
    in_vals = list(in_degrees.values())
    q75 = np.percentile(in_vals, 75) if in_vals else 1
    q25 = np.percentile(in_vals, 25) if in_vals else 0
    node_colors = [
        PALETTE["secondary"] if in_degrees[n] >= q75
        else PALETTE["primary"]  if in_degrees[n] >= q25
        else PALETTE["muted"]
        for n in G_sub.nodes()
    ]

    try:
        pos = nx.spring_layout(G_sub, seed=42, k=1.5 / np.sqrt(len(G_sub.nodes())))
    except Exception:
        pos = nx.random_layout(G_sub, seed=42)

    nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(G_sub, pos, ax=ax, edge_color=PALETTE["muted"],
                           alpha=0.2, arrows=False, width=0.5)
    ax.set_title(f"{title}\n(top {len(G_sub.nodes())} nodes by degree)")
    ax.axis("off")


def plot_social_vs_interactions(G: nx.DiGraph, inter_df: pd.DataFrame, ax: plt.Axes):
    social_degrees     = dict(G.degree())
    interaction_counts = inter_df["user_id"].value_counts().to_dict()
    common_users = set(social_degrees.keys()) & set(interaction_counts.keys())
    if not common_users:
        ax.text(0.5, 0.5, "No common users between .inter and .net",
                transform=ax.transAxes, ha="center", fontsize=9)
        return

    x = [social_degrees[u]     for u in common_users]
    y = [interaction_counts[u] for u in common_users]
    ax.scatter(x, y, alpha=0.35, s=12, color=PALETTE["primary"], rasterized=True, zorder=2)

    if len(x) > 5:
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), color=PALETTE["secondary"],
                linewidth=1.5, linestyle="--", label=f"Trend (slope={z[0]:.2f})", zorder=3)
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.97, 0.05, f"r = {corr:.3f}", transform=ax.transAxes,
                ha="right", fontsize=9, color=PALETTE["muted"])

    ax.set_title("Social Degree vs. Interaction Count per User")
    ax.set_xlabel("Social Degree (edges in .net)")
    ax.set_ylabel("Interaction Count (rows in .inter)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)


SYNTH_TYPES = [
    "contrarian", "echo_chamber", "fake_users", "line_graph",
    "partial_alignment", "random", "scale_free", "small_world", "star",
]

def _classify_topology(name: str) -> str:
    for t in SYNTH_TYPES:
        if name.startswith(t):
            return t
    return "real"


def compute_summary(inter_df: pd.DataFrame, G: Optional[nx.DiGraph],
                    dataset_name: str, dseed: int = 0) -> dict:
    """Return a flat dict of numeric stats — suitable for both display and CSV."""
    n_users = inter_df["user_id"].nunique()
    n_items = inter_df["item_id"].nunique()
    n_inter = len(inter_df)
    density = n_inter / (n_users * n_items)

    user_counts = inter_df["user_id"].value_counts()
    item_counts = inter_df["item_id"].value_counts()

    params = parse_dataset_params(dataset_name)
    n_items_catalog = int(params.get("n_items", 0))
    catalog_coverage = n_items / n_items_catalog if n_items_catalog > 0 else float("nan")

    row = {
        # identity
        "dataset":           dataset_name,
        "dseed":             dseed,
        "topology":          _classify_topology(dataset_name),
        # parsed params
        "param_n_users":     params.get("n_users",   float("nan")),
        "param_n_items":     params.get("n_items",   float("nan")),
        "param_sparsity":    params.get("sparsity",  float("nan")),
        "param_noise":       params.get("noise",     float("nan")),
        "param_fake":        params.get("fake",      float("nan")),
        "param_sd":          params.get("sd",        float("nan")),
        "param_deg":         params.get("deg",       float("nan")),
        # interaction stats
        "n_interactions":    n_inter,
        "n_users_actual":    n_users,
        "n_items_actual":    n_items,
        "density":           density,
        "catalog_coverage":  catalog_coverage,
        "gini_user":         _gini(user_counts.values),
        "gini_item":         _gini(item_counts.values),
        "user_inter_mean":   float(user_counts.mean()),
        "user_inter_std":    float(user_counts.std()),
        "user_inter_median": float(user_counts.median()),
        "item_inter_mean":   float(item_counts.mean()),
        "item_inter_std":    float(item_counts.std()),
        "item_inter_median": float(item_counts.median()),
        # social graph (filled below if .net present)
        "graph_nodes":       float("nan"),
        "graph_edges":       float("nan"),
        "avg_social_degree": float("nan"),
        "max_social_degree": float("nan"),
        "clustering":        float("nan"),
        "lcc_size":          float("nan"),
    }

    if G is not None:
        degrees = [d for _, d in G.degree()]
        row["graph_nodes"]       = G.number_of_nodes()
        row["graph_edges"]       = G.number_of_edges()
        row["avg_social_degree"] = float(np.mean(degrees))
        row["max_social_degree"] = float(max(degrees))
        try:
            if G.number_of_nodes() < 20000:
                ug = G.to_undirected()
                row["clustering"] = nx.average_clustering(ug)
                lcc = max(nx.connected_components(ug), key=len)
                row["lcc_size"]   = len(lcc)
        except Exception:
            pass

    return row


def _gini(arr: np.ndarray) -> float:
    arr = np.sort(arr.astype(float))
    n = len(arr)
    if n == 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n + 1) * arr) / (n * arr.sum()) - (n + 1) / n)


def print_summary(row: dict):
    print(f"\n{'='*55}")
    print(f"  Dataset: {row['dataset']}")
    print(f"{'='*55}")
    print(f"  Interactions  : {row['n_interactions']:,}")
    print(f"  Users         : {row['n_users_actual']:,}")
    print(f"  Items         : {row['n_items_actual']:,}")
    print(f"  Density       : {row['density']:.4%}")
    if not np.isnan(row["catalog_coverage"]):
        print(f"  Coverage      : {row['catalog_coverage']:.1%}")
    print(f"  Gini (users)  : {row['gini_user']:.4f}")
    print(f"  Gini (items)  : {row['gini_item']:.4f}")
    params = {k[6:]: v for k, v in row.items()
              if k.startswith("param_") and not np.isnan(v)}
    if params:
        print(f"  Params        : {params}")
    if not np.isnan(row["graph_nodes"]):
        print(f"\n  Social graph")
        print(f"  Nodes         : {int(row['graph_nodes']):,}")
        print(f"  Edges         : {int(row['graph_edges']):,}")
        print(f"  Avg degree    : {row['avg_social_degree']:.2f}")
        print(f"  Max degree    : {int(row['max_social_degree'])}")
        if not np.isnan(row["clustering"]):
            print(f"  Clustering    : {row['clustering']:.4f}")
        if not np.isnan(row["lcc_size"]):
            print(f"  LCC size      : {int(row['lcc_size']):,}")
    print(f"{'='*55}\n")


def save_stats_file(row: dict, path: str):
    """Write human-readable stats.txt for a single dataset."""
    with open(path, "w") as f:
        for k, v in row.items():
            f.write(f"{k}: {v}\n")


def append_stats_csv(row: dict, csv_path: str):
    """Append one dataset row, aligning columns with any existing CSV (topologies have disjoint key sets)."""
    df_row = pd.DataFrame([row])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df_row], ignore_index=True)
    else:
        combined = df_row
    combined.to_csv(csv_path, index=False)


def process_dataset(dataset_name: str, inter_path: str, net_path: Optional[str],
                    out_dir: str, graph_sample: int, dseed: int = 0,
                    stats_csv: Optional[str] = None):
    ds_out = os.path.join(out_dir, dataset_name)
    os.makedirs(ds_out, exist_ok=True)

    already_done = os.path.exists(os.path.join(ds_out, "interactions.png"))

    # always load data so we can write the CSV row (even for already-visualized datasets)
    inter_df = load_inter(inter_path)
    has_rating = "rating" in inter_df.columns

    G = None
    if net_path and os.path.exists(net_path):
        net_df = load_net(net_path)
        G = build_nx_graph(net_df)

    row = compute_summary(inter_df, G, dataset_name, dseed=dseed)

    # merge diagnostics.json if present (cosine + Jaccard signal gaps)
    diag_path = os.path.join(os.path.dirname(inter_path), "diagnostics.json")
    if os.path.exists(diag_path):
        import json
        with open(diag_path) as f:
            diag = json.load(f)
        # pull in all scalar diagnostics (skip lists like group_sizes/community_sizes_top5)
        for key, val in diag.items():
            if key in ("dataset", "seed", "topology"):
                continue  # already in row
            if isinstance(val, (int, float)):
                row[key] = val

    if stats_csv:
        append_stats_csv(row, stats_csv)

    if already_done:
        print(f"  Skipping '{dataset_name}' — already visualized (stats recorded).")
        return

    print(f"\n>>> Processing dataset: {dataset_name} <<<")
    print_summary(row)
    save_stats_file(row, os.path.join(ds_out, "stats.txt"))

    # figure 1: interaction statistics
    print("    Plotting Figure 1: Interaction statistics...")
    n_inter_plots = 3 if has_rating else 2
    fig1, axes1 = plt.subplots(1, n_inter_plots, figsize=(5 * n_inter_plots, 4.5))
    fig1.suptitle(f"{dataset_name} — Interaction Statistics", fontsize=13, fontweight="bold", y=1.02)
    if n_inter_plots == 1:
        axes1 = [axes1]
    plot_interaction_overview(inter_df, list(axes1), has_rating)
    fig1.tight_layout()
    fig1.savefig(os.path.join(ds_out, "interactions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # figure 2: item popularity CCDF + catalog coverage
    print("    Plotting Figure 2: Popularity analysis...")
    fig2, (ax_ccdf, ax_cov) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig2.suptitle(f"{dataset_name} — Item Popularity", fontsize=13, fontweight="bold", y=1.02)
    plot_popularity_analysis(inter_df, dataset_name, ax_ccdf, ax_cov)
    fig2.tight_layout()
    fig2.savefig(os.path.join(ds_out, "popularity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    if G is not None:
        print("    Plotting Figure 3: Degree distributions...")
        fig3, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(11, 4.5))
        fig3.suptitle(f"{dataset_name} — Social Graph Degree Distributions",
                      fontsize=13, fontweight="bold", y=1.02)
        plot_degree_distribution(G, ax_in, ax_out)
        fig3.tight_layout()
        fig3.savefig(os.path.join(ds_out, "degree_dist.png"), dpi=150, bbox_inches="tight")
        plt.close(fig3)

        print(f"    Plotting Figure 4: Graph sample (top {graph_sample} nodes)...")
        fig4, ax4 = plt.subplots(figsize=(9, 9))
        fig4.suptitle(f"{dataset_name} — Social Graph", fontsize=13, fontweight="bold")
        plot_graph_sample(G, ax4, n_sample=graph_sample, title=dataset_name)
        fig4.tight_layout()
        fig4.savefig(os.path.join(ds_out, "graph.png"), dpi=150, bbox_inches="tight")
        plt.close(fig4)

        print("    Plotting Figure 5: Social vs interaction cross statistics...")
        fig5, ax5 = plt.subplots(figsize=(6.5, 4.5))
        fig5.suptitle(f"{dataset_name} — Social ↔ Interaction Cross Analysis",
                      fontsize=13, fontweight="bold", y=1.02)
        plot_social_vs_interactions(G, inter_df, ax5)
        fig5.tight_layout()
        fig5.savefig(os.path.join(ds_out, "cross.png"), dpi=150, bbox_inches="tight")
        plt.close(fig5)


def main():
    parser = argparse.ArgumentParser(description="Batch Visualize RecBole .inter and .net files")
    parser.add_argument("--datasets_dir", type=str, default=None,
                        help="Path to datasets directory (overrides --dseed)")
    parser.add_argument("--dseed",        type=int, default=None,
                        help="Data seed — shorthand for --datasets_dir ./datasets/dseed_N")
    parser.add_argument("--out",          type=str, default="./exploration/dataset_graphs")
    parser.add_argument("--graph_sample", type=int, default=300)
    args = parser.parse_args()

    # resolve datasets directory
    if args.datasets_dir:
        datasets_dir = args.datasets_dir
    elif args.dseed is not None:
        datasets_dir = f"./datasets/dseed_{args.dseed}"
    else:
        datasets_dir = "./datasets"

    os.makedirs(args.out, exist_ok=True)
    stats_csv = os.path.join(args.out, "dataset_stats.csv")
    log_path  = os.path.join(args.out, "viz_log.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    dseed_val = args.dseed if args.dseed is not None else 0

    print(f"Datasets directory : {datasets_dir}")
    print(f"Output directory   : {args.out}")
    print(f"Stats CSV          : {stats_csv}\n")

    if not os.path.exists(datasets_dir):
        print(f"Error: datasets directory '{datasets_dir}' not found.")
        sys.stdout = tee.terminal
        tee.close()
        return

    datasets_processed = 0
    for folder_name in sorted(os.listdir(datasets_dir)):
        folder_path = os.path.join(datasets_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        inter_path = os.path.join(folder_path, f"{folder_name}.inter")
        net_path   = os.path.join(folder_path, f"{folder_name}.net")

        if os.path.exists(inter_path):
            process_dataset(
                dataset_name=folder_name,
                inter_path=inter_path,
                net_path=net_path,
                out_dir=args.out,
                graph_sample=args.graph_sample,
                dseed=dseed_val,
                stats_csv=stats_csv,
            )
            datasets_processed += 1
        else:
            print(f"\nSkipping '{folder_name}' — no .inter file found.")

    print(f"\nDone! Processed {datasets_processed} datasets.")
    print(f"Plots saved to:  {args.out}/<dataset_name>/")
    print(f"Log saved to:    {log_path}")

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
