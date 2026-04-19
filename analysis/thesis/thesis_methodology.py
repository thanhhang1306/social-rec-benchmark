"""Figures for Chapter 3: methodology."""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import MaxNLocator, ScalarFormatter

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BASE = "/n/fs/recbench/new_rec/datasets"
OUT = "/n/fs/recbench/new_rec/figures/methodology"

TOPOLOGIES = [
    ("scale_free", "Scale-Free", "I"),
    ("small_world", "Small-World", "I"),
    ("random", "Random (ER)", "I"),
    ("echo_chamber", "Echo Chamber", "II"),
    ("partial_alignment", "Partial Align.", "II"),
    ("contrarian", "Contrarian", "II"),
    ("star", "Star", "III"),
    ("line_graph", "Line Graph", "III"),
    ("fake_users", "Fake Users", "III"),
]

GROUP_COLORS = {"I": "#0072B2", "II": "#E69F00", "III": "#009E73"}
GROUP_LABELS = {
    "I": "Group I:\nStructure",
    "II": "Group II:\nHomophily",
    "III": "Group III:\nStructural Integrity",
}

SEEDS = [1, 2, 3]


def _save(fig, filename):
    """Save fig as both PDF and PNG in OUT."""
    pdf_path = os.path.join(OUT, filename)
    fig.savefig(pdf_path)
    png_path = pdf_path.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=200)
    print("wrote " + pdf_path.replace(".pdf", ".{pdf,png}"))


def dataset_name(topo_key):
    if topo_key == "line_graph":
        return f"{topo_key}_u2000_i5000_sp0.01_ns0.1_max30_decay0.5"
    elif topo_key == "fake_users":
        return f"{topo_key}_u2000_i5000_sp0.01_ns0.1_fake0.1"
    else:
        return f"{topo_key}_u2000_i5000_sp0.01_ns0.1"


def load_net_degrees(seed, topo_key):
    """Load social network, return undirected degree sequence as array."""
    ds = dataset_name(topo_key)
    path = os.path.join(BASE, f"dseed_{seed}", ds, f"{ds}.net")
    df = pd.read_csv(path, sep="\t")
    all_nodes = pd.concat([df[df.columns[0]], df[df.columns[1]]])
    return all_nodes.value_counts().values


def load_inter_counts(seed, topo_key):
    """Load interaction file, return per-user counts as array."""
    ds = dataset_name(topo_key)
    path = os.path.join(BASE, f"dseed_{seed}", ds, f"{ds}.inter")
    df = pd.read_csv(path, sep="\t")
    return df[df.columns[0]].value_counts().values


def compute_all_stats(seed, topo_key):
    """Compute all stats for one topology at one seed."""
    ds = dataset_name(topo_key)
    inter_path = os.path.join(BASE, f"dseed_{seed}", ds, f"{ds}.inter")
    net_path = os.path.join(BASE, f"dseed_{seed}", ds, f"{ds}.net")

    inter = pd.read_csv(inter_path, sep="\t")
    net = pd.read_csv(net_path, sep="\t")

    n_users = inter[inter.columns[0]].nunique()
    n_items = inter[inter.columns[1]].nunique()
    n_inter = len(inter)

    deg = load_net_degrees(seed, topo_key)

    G = nx.Graph()
    for _, row in net.iterrows():
        G.add_edge(row.iloc[0], row.iloc[1])
    clustering = nx.average_clustering(G)

    uc = inter[inter.columns[0]].value_counts().values
    uc_sorted = np.sort(uc)
    n = len(uc_sorted)
    gini = (2 * np.sum(np.arange(1, n + 1) * uc_sorted) / (n * np.sum(uc_sorted))) - (n + 1) / n

    return {
        "n_inter": n_inter,
        "density": n_inter / (n_users * n_items),
        "avg_degree": deg.mean(),
        "max_degree": deg.max(),
        "clustering": clustering,
        "user_gini": gini,
        "n_edges": len(net),
        "n_users": n_users,
        "n_items": n_items,
    }


def fig_degree_distributions():
    fig, axes = plt.subplots(3, 3, figsize=(6.5, 5.8))

    for idx, (topo_key, label, group) in enumerate(TOPOLOGIES):
        ax = axes[idx // 3][idx % 3]
        deg = load_net_degrees(1, topo_key)
        color = GROUP_COLORS[group]

        if topo_key == "star":
            # star degree is two-point (hub vs leaves) — labeled bars read better than a histogram
            counts = Counter(deg)
            degrees = sorted(counts.keys())
            freqs = [counts[d] for d in degrees]
            bars = ax.bar(range(len(degrees)), freqs, color=color, alpha=0.8, edgecolor="none")
            ax.set_xticks(range(len(degrees)))
            ax.set_xticklabels([str(d) for d in degrees])
            # label bars with counts
            for bar, freq in zip(bars, freqs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20,
                    str(freq),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            ax.set_ylim(0, max(freqs) * 1.15)

        elif topo_key == "line_graph":
            # line graph: nearly all degree 2, endpoints degree 1
            counts = Counter(deg)
            degrees = sorted(counts.keys())
            freqs = [counts[d] for d in degrees]
            ax.bar(degrees, freqs, width=0.6, color=color, alpha=0.8, edgecolor="none")
            ax.set_xlim(-0.5, max(degrees) + 0.5)

        elif topo_key == "scale_free":
            # scale-free: use log-log to show power law
            counts = Counter(deg)
            degrees = sorted(counts.keys())
            freqs = [counts[d] for d in degrees]
            ax.scatter(degrees, freqs, s=12, color=color, alpha=0.8, zorder=3)
            # power-law reference line
            d_arr = np.array(degrees, dtype=float)
            f_arr = np.array(freqs, dtype=float)
            mask = d_arr >= 2
            if mask.sum() > 2:
                coeffs = np.polyfit(np.log10(d_arr[mask]), np.log10(f_arr[mask]), 1)
                x_fit = np.linspace(np.log10(2), np.log10(max(degrees)), 50)
                ax.plot(
                    10**x_fit,
                    10 ** np.polyval(coeffs, x_fit),
                    "--",
                    color="grey",
                    alpha=0.5,
                    linewidth=0.8,
                    label=f"slope={coeffs[0]:.1f}",
                )
                ax.legend(loc="upper right", frameon=False, fontsize=6)
            ax.set_xscale("log")
            ax.set_yscale("log")

        else:
            # standard histogram for ER-like distributions
            counts = Counter(deg)
            degrees = sorted(counts.keys())
            freqs = [counts[d] for d in degrees]
            ax.bar(degrees, freqs, width=0.8, color=color, alpha=0.8, edgecolor="none")
            # don't use log scale if the range is moderate
            if max(freqs) / max(min(freqs), 1) > 100:
                ax.set_yscale("log")

        ax.set_title(f"{label}", fontsize=8.5, pad=4, fontweight="medium")
        ax.set_xlabel("Degree")
        if idx % 3 == 0:
            ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    for row, (gkey, glabel) in enumerate(
        [("I", "Group I"), ("II", "Group II"), ("III", "Group III")]
    ):
        axes[row][2].annotate(
            glabel,
            xy=(1.08, 0.5),
            xycoords="axes fraction",
            fontsize=8,
            ha="left",
            va="center",
            color=GROUP_COLORS[gkey],
            fontweight="bold",
            rotation=-90,
        )

    plt.tight_layout(w_pad=1.0, h_pad=1.2)
    _save(fig, "degree_distributions.pdf")
    plt.close(fig)


def fig_topology_comparison():
    """Horizontal dot plot with zoomed axes and outlier annotations."""

    # gather stats across seeds
    stats = {
        m: {t[0]: [] for t in TOPOLOGIES}
        for m in ["density", "avg_degree", "clustering", "user_gini"]
    }

    for seed in SEEDS:
        for topo_key, _, _ in TOPOLOGIES:
            s = compute_all_stats(seed, topo_key)
            stats["density"][topo_key].append(s["density"])
            stats["avg_degree"][topo_key].append(s["avg_degree"])
            stats["clustering"][topo_key].append(s["clustering"])
            stats["user_gini"][topo_key].append(s["user_gini"])

    labels = [t[1] for t in TOPOLOGIES]
    groups = [t[2] for t in TOPOLOGIES]
    colors = [GROUP_COLORS[g] for g in groups]
    y = np.arange(len(TOPOLOGIES))

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))

    panels = [
        ("density", "Interaction Density", axes[0, 0]),
        ("avg_degree", "Avg Social Degree", axes[0, 1]),
        ("clustering", "Avg Clustering Coeff.", axes[1, 0]),
        ("user_gini", "User Interaction Gini", axes[1, 1]),
    ]

    for metric, title, ax in panels:
        means = np.array([np.mean(stats[metric][t[0]]) for t in TOPOLOGIES])
        stds = np.array([np.std(stats[metric][t[0]]) for t in TOPOLOGIES])

        # draw horizontal dots with error bars
        for i in range(len(TOPOLOGIES)):
            ax.errorbar(
                means[i],
                y[i],
                xerr=stds[i],
                fmt="o",
                color=colors[i],
                markersize=5,
                capsize=3,
                capthick=0.8,
                linewidth=0.8,
                zorder=3,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(title, fontsize=8.5)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)

        # zoom x-axis to show variation, handle outliers with annotations
        if metric == "density":
            # standard topologies cluster near 0.011; line graph at ~0.004
            standard = [m for m, t in zip(means, TOPOLOGIES) if t[0] != "line_graph"]
            xmin = min(standard) * 0.97
            xmax = max(standard) * 1.03
            ax.set_xlim(xmin, xmax)
            # mark line graph as off-scale
            lg_idx = [t[0] for t in TOPOLOGIES].index("line_graph")
            ax.plot(
                xmin + (xmax - xmin) * 0.02,
                y[lg_idx],
                "<",
                color=GROUP_COLORS["III"],
                markersize=6,
                zorder=4,
            )
            ax.annotate(
                f"{means[lg_idx]:.4f} \u2190",
                xy=(xmin + (xmax - xmin) * 0.04, y[lg_idx]),
                fontsize=6.5,
                va="center",
                color=GROUP_COLORS["III"],
                style="italic",
            )

        elif metric == "avg_degree":
            # range ~4 to 15, show full range but start near 0
            ax.set_xlim(0, max(means) * 1.1)

        elif metric == "clustering":
            # small-world ~0.44, scale-free ~0.02, rest ~0.003 — log scale spans the 100x range
            ax.set_xscale("log")
            nonzero = means[means > 0]
            if len(nonzero) > 0:
                ax.set_xlim(min(nonzero) * 0.5, max(nonzero) * 2)
            for i, m in enumerate(means):
                if m < 1e-6:
                    ax.annotate(
                        "0",
                        xy=(ax.get_xlim()[0] * 1.5, y[i]),
                        fontsize=6,
                        va="center",
                        ha="left",
                        color=colors[i],
                        style="italic",
                    )

        elif metric == "user_gini":
            all_vals = means
            margin = (max(all_vals) - min(all_vals)) * 0.3
            ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin)

        ax.axhline(y=2.5, color="grey", linewidth=0.3, linestyle="--")
        ax.axhline(y=5.5, color="grey", linewidth=0.3, linestyle="--")

    plt.tight_layout(w_pad=1.5, h_pad=1.0)
    _save(fig, "topology_comparison.pdf")
    plt.close(fig)


def fig_seed_reproducibility():
    """Grouped horizontal dot plot showing CV across seeds for key metrics."""

    metrics = ["Interaction Count", "Avg Degree"]
    metric_keys = ["n_inter", "avg_degree"]

    raw = {mk: {t[0]: [] for t in TOPOLOGIES} for mk in metric_keys}
    for seed in SEEDS:
        for topo_key, _, _ in TOPOLOGIES:
            s = compute_all_stats(seed, topo_key)
            for mk in metric_keys:
                raw[mk][topo_key].append(s[mk])

    labels = [t[1] for t in TOPOLOGIES]
    groups = [t[2] for t in TOPOLOGIES]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5, 3.0), sharey=True)

    y = np.arange(len(TOPOLOGIES))

    for ax, metric_name, mk in zip(axes, metrics, metric_keys):
        cvs = []
        for topo_key, _, _ in TOPOLOGIES:
            vals = np.array(raw[mk][topo_key])
            cv = np.std(vals) / np.mean(vals) * 100 if np.mean(vals) != 0 else 0
            cvs.append(cv)

        for i in range(len(TOPOLOGIES)):
            ax.barh(y[i], cvs[i], height=0.65, color=colors[i], alpha=0.8, edgecolor="none")

        ax.set_xlabel("CV (%)", fontsize=7.5)
        ax.set_title(metric_name, fontsize=8.5, pad=4)
        ax.set_xlim(0, max(max(cvs) * 1.2, 0.5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        # group separators
        ax.axhline(y=2.5, color="grey", linewidth=0.3, linestyle="--")
        ax.axhline(y=5.5, color="grey", linewidth=0.3, linestyle="--")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()

    plt.tight_layout()
    _save(fig, "seed_reproducibility.pdf")
    plt.close(fig)


def _load_inter(seed, topo_key):
    ds = dataset_name(topo_key)
    path = os.path.join(BASE, f"dseed_{seed}", ds, f"{ds}.inter")
    return pd.read_csv(path, sep="\t")


def fig_rating_distribution():
    """Histogram of rating values pooled across Group I topologies at data seed 1."""
    ratings = []
    for topo_key, _, group in TOPOLOGIES:
        if group != "I":
            continue
        df = _load_inter(1, topo_key)
        col = [c for c in df.columns if c.startswith("rating")][0]
        ratings.append(df[col].values)
    ratings = np.concatenate(ratings)

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bins = np.arange(0.5, 6.5, 1.0)
    ax.hist(ratings, bins=bins, color="#0072B2", edgecolor="white", linewidth=0.6, rwidth=0.85)
    mean = float(ratings.mean())
    ax.axvline(mean, color="#D55E00", linestyle="--", linewidth=1.0, label=f"Mean = {mean:.2f}")
    ax.set_xlabel("Rating value")
    ax.set_ylabel("Interaction count")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    _save(fig, "rating_distribution.pdf")
    print(f"    (mean = {mean:.3f}, N = {len(ratings):,})")
    plt.close(fig)


def fig_interaction_distributions():
    """Per-user interaction count histograms, one panel per topology group."""
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True)
    groups = ["I", "II", "III"]

    # a shared x-range keeps panels comparable
    all_counts = []
    per_topo = {}
    for topo_key, label, g in TOPOLOGIES:
        try:
            df = _load_inter(1, topo_key)
        except FileNotFoundError:
            continue
        c = df.groupby(df.columns[0]).size().values
        per_topo[topo_key] = (label, g, c)
        all_counts.append(c)
    pooled = np.concatenate(all_counts)
    x_max = int(np.percentile(pooled, 99.5))
    bins = np.linspace(0, x_max, 45)

    for ax, g in zip(axes, groups):
        for topo_key, label, gg in TOPOLOGIES:
            if gg != g or topo_key not in per_topo:
                continue
            _, _, counts = per_topo[topo_key]
            ax.hist(counts, bins=bins, histtype="step", linewidth=1.3, label=label)
        ax.set_yscale("log")
        ax.set_xlim(0, x_max)
        ax.set_title(f"{GROUP_LABELS[g].replace(chr(10), ' ')}", fontsize=9)
        ax.set_xlabel("Interactions per user")
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    axes[0].set_ylabel("User count (log)")
    plt.tight_layout()
    _save(fig, "interaction_distributions.pdf")
    plt.close(fig)


def _bfs_two_color(G, n):
    color = np.full(n, -1, dtype=int)
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb in G.neighbors(node):
                if color[nb] == -1:
                    color[nb] = 1 - color[node]
                    queue.append(nb)
    color[color == -1] = 0
    return color


def fig_contrarian_impurity():
    """Sweep ER average degree, compute impurity under BFS 2-coloring."""
    n_users = 2000
    k_values = [2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 75, 100]
    n_trials = 3

    means, stds = [], []
    for k_bar in k_values:
        p = k_bar / (n_users - 1)
        trial_imp = []
        for t in range(n_trials):
            G = nx.erdos_renyi_graph(n_users, p, seed=100 + t)
            color = _bfs_two_color(G, n_users)
            edges = list(G.edges())
            if not edges:
                trial_imp.append(0.0)
                continue
            cross = sum(1 for u, v in edges if color[u] != color[v])
            impurity = 1 - cross / len(edges)
            trial_imp.append(impurity)
        means.append(np.mean(trial_imp))
        stds.append(np.std(trial_imp))

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.plot(
        k_values,
        means,
        "-o",
        color="#E69F00",
        markersize=4,
        linewidth=1.2,
        label="BFS 2-coloring impurity",
    )
    ax.fill_between(k_values, means - stds, means + stds, color="#E69F00", alpha=0.2)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="Random assignment (0.5)")
    ax.set_xlabel("Average social degree $\\bar{k}$")
    ax.set_ylabel("Within-group edge fraction")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    y_hi = max(0.65, float(np.max(means + stds)) * 1.08)
    ax.set_ylim(0, y_hi)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    plt.tight_layout()
    _save(fig, "contrarian_impurity.pdf")
    plt.close(fig)


import random as stdlib_random


def _sample_jaccard(inter_df, net_df, sample_size=500, seed=0):
    stdlib_random.seed(seed)
    user_col = inter_df.columns[0]
    item_col = inter_df.columns[1]
    user_items = inter_df.groupby(user_col)[item_col].apply(set).to_dict()

    src_col, tgt_col = net_df.columns[0], net_df.columns[1]
    edges = [
        (u, v)
        for u, v in zip(net_df[src_col], net_df[tgt_col])
        if u in user_items and v in user_items and u != v
    ]
    if not edges:
        return np.array([]), np.array([])

    sampled = stdlib_random.sample(edges, min(sample_size, len(edges)))
    edge_jac = []
    for u, v in sampled:
        su, sv = user_items[u], user_items[v]
        union = len(su | sv)
        edge_jac.append(len(su & sv) / union if union > 0 else 0.0)

    users = list(user_items.keys())
    rand_jac = []
    for _ in range(sample_size):
        u = stdlib_random.choice(users)
        v = stdlib_random.choice(users)
        if u == v:
            continue
        su, sv = user_items[u], user_items[v]
        union = len(su | sv)
        rand_jac.append(len(su & sv) / union if union > 0 else 0.0)

    return np.array(edge_jac), np.array(rand_jac)


def fig_jaccard_histograms():
    """Compare per-pair Jaccard overlap for a synthetic reference and three real datasets."""
    panels = [
        (
            "Scale-Free (synth.)",
            os.path.join(
                BASE, "dseed_1", dataset_name("scale_free"), f"{dataset_name('scale_free')}.inter"
            ),
            os.path.join(
                BASE, "dseed_1", dataset_name("scale_free"), f"{dataset_name('scale_free')}.net"
            ),
        ),
        ("Yelp", os.path.join(BASE, "yelp", "yelp.inter"), os.path.join(BASE, "yelp", "yelp.net")),
        (
            "LastFM",
            os.path.join(BASE, "lastfm", "lastfm.inter"),
            os.path.join(BASE, "lastfm", "lastfm.net"),
        ),
        (
            "Douban-Book",
            os.path.join(BASE, "douban-book", "douban-book.inter"),
            os.path.join(BASE, "douban-book", "douban-book.net"),
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    flat_axes = axes.flatten()

    for ax, (label, inter_path, net_path) in zip(flat_axes, panels):
        try:
            inter_df = pd.read_csv(inter_path, sep="\t")
            net_df = pd.read_csv(net_path, sep="\t")
        except FileNotFoundError as e:
            print(f"  [skip] {label}: {e}")
            ax.set_title(f"{label} (missing)")
            continue
        edge_j, rand_j = _sample_jaccard(inter_df, net_df, sample_size=500, seed=42)
        if len(edge_j) == 0:
            ax.set_title(f"{label} (no data)")
            continue
        # per-panel x-range: accommodate 99th percentile of connected-pair Jaccard
        x_hi = max(float(np.percentile(edge_j, 99)) * 1.15, 0.03)
        bins = np.linspace(0, x_hi, 35)
        ax.hist(
            rand_j,
            bins=bins,
            color="#999999",
            alpha=0.65,
            label=f"Random (mean {rand_j.mean():.3f})",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.hist(
            edge_j,
            bins=bins,
            color="#D55E00",
            alpha=0.65,
            label=f"Connected (mean {edge_j.mean():.3f})",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlim(0, x_hi)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Jaccard similarity")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        print(
            f"  [{label}] random={rand_j.mean():.4f}  connected={edge_j.mean():.4f}  gap={edge_j.mean() - rand_j.mean():+.4f}"
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("Pair count")
    plt.tight_layout()
    _save(fig, "jaccard_histograms.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating methodology figures...")
    print()
    print("[1/7] Degree distributions")
    fig_degree_distributions()
    print("[2/7] Topology comparison")
    fig_topology_comparison()
    print("[3/7] Seed reproducibility")
    fig_seed_reproducibility()
    print("[4/7] Rating distribution")
    fig_rating_distribution()
    print("[5/7] Interaction distributions")
    fig_interaction_distributions()
    print("[6/7] Contrarian impurity sweep")
    fig_contrarian_impurity()
    print("[7/7] Jaccard histograms")
    fig_jaccard_histograms()
    print()
    print(f"All figures saved to {OUT}/")
