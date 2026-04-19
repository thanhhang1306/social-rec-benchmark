"""Figures for Chapter 3: topology-group schematic diagrams."""

import os
import random

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

OUT = "/n/fs/recbench/new_rec/figures/background"
SEED = 7
N = 60

ANN_FONTSIZE = 11


def _draw(
    ax,
    G,
    pos,
    node_color,
    *,
    edgecolors="black",
    linewidths=0.5,
    node_size=140,
    edge_color="#bbbbbb",
    edge_width=0.5,
    cmap=None,
):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=edge_width)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_color,
        cmap=cmap,
        node_size=node_size,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    ax.set_axis_off()


def _degree_style(G):
    degs = np.array([G.degree(n) for n in G.nodes()], dtype=float)
    sizes = 60 + 320 * (degs - degs.min()) / max(1.0, (degs.max() - degs.min()))
    return degs, sizes




def group1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # scale-free
    G = nx.barabasi_albert_graph(N, 3, seed=SEED)
    pos = nx.spring_layout(G, seed=SEED)
    degs, sizes = _degree_style(G)
    _draw(axes[0], G, pos, degs, cmap="YlOrRd", node_size=sizes)
    axes[0].set_title(f"Scale-Free (BA, $m=3$)\nClustering: {nx.average_clustering(G):.3f}")

    # small-world
    G = nx.watts_strogatz_graph(N, 6, 0.1, seed=SEED)
    pos = nx.spring_layout(G, seed=SEED)
    degs, sizes = _degree_style(G)
    _draw(axes[1], G, pos, degs, cmap="YlOrRd", node_size=sizes)
    axes[1].set_title(f"Small-World (WS, $k=6, p=0.1$)\nClustering: {nx.average_clustering(G):.3f}")

    # random ER
    p_er = 6.0 / (N - 1)
    G = nx.erdos_renyi_graph(N, p_er, seed=SEED)
    pos = nx.spring_layout(G, seed=SEED)
    degs, sizes = _degree_style(G)
    _draw(axes[2], G, pos, degs, cmap="YlOrRd", node_size=sizes)
    axes[2].set_title(r"Random (ER, $\bar{k}=6$)" + f"\nClustering: {nx.average_clustering(G):.3f}")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "group1_topologies.png"))
    plt.close(fig)
    print("wrote group1_topologies.png")




def group2():
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    p_er = 6.0 / (N - 1)
    G = nx.erdos_renyi_graph(N, p_er, seed=SEED)
    pos = nx.spring_layout(G, seed=SEED)

    # echo chamber: communities (color = community)
    communities = list(greedy_modularity_communities(G))
    palette = plt.get_cmap("tab10")
    assignment = np.zeros(N, dtype=int)
    for ci, comm in enumerate(communities):
        for u in comm:
            assignment[u] = ci
    comm_colors = np.array([palette(assignment[u] % 10) for u in range(N)])
    _draw(axes[0], G, pos, [tuple(c) for c in comm_colors])
    axes[0].set_title("Echo Chamber\n$\\cos = 1$ within community")

    # partial alignment: border = community, fill = personal component (random per user)
    personal_colors = np.array([palette(rng.integers(0, 10)) for _ in range(N)])
    nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color="#bbbbbb", width=0.5)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=axes[1],
        node_color=[tuple(c) for c in personal_colors],
        node_size=180,
        edgecolors=[tuple(c) for c in comm_colors],
        linewidths=2.2,
    )
    axes[1].set_axis_off()
    axes[1].set_title(
        "Partial Alignment ($d_s = 5$)\n$\\mathbb{E}[\\cos] \\approx 0.5$ within community"
    )

    # contrarian: BFS 2-coloring; cross-group edges in red
    color_arr = np.full(N, -1, dtype=int)
    for start in range(N):
        if color_arr[start] != -1:
            continue
        color_arr[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb in G.neighbors(node):
                if color_arr[nb] == -1:
                    color_arr[nb] = 1 - color_arr[node]
                    queue.append(nb)
    color_arr[color_arr == -1] = 0
    same_edges = [(u, v) for u, v in G.edges() if color_arr[u] == color_arr[v]]
    cross_edges = [(u, v) for u, v in G.edges() if color_arr[u] != color_arr[v]]
    nx.draw_networkx_edges(G, pos, ax=axes[2], edgelist=same_edges, edge_color="#bbbbbb", width=0.5)
    nx.draw_networkx_edges(
        G, pos, ax=axes[2], edgelist=cross_edges, edge_color="#d62728", width=1.1
    )
    node_colors = ["#d62728" if c == 0 else "#1f77b4" for c in color_arr]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=axes[2],
        node_color=node_colors,
        node_size=140,
        edgecolors="black",
        linewidths=0.5,
    )
    axes[2].set_axis_off()
    axes[2].set_title("Contrarian (BFS 2-coloring)\n$\\mathbb{E}[\\cos] \\approx -1$ across groups")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "group2_topologies.png"))
    plt.close(fig)
    print("wrote group2_topologies.png")




def group3():
    rng = np.random.default_rng(SEED)  # noqa: F841
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # star
    n_star = 40
    G = nx.star_graph(n_star - 1)
    pos = nx.circular_layout(G)
    pos[0] = np.array([0.0, 0.0])  # hub at center
    colors = ["#d62728" if n == 0 else "#4C72B0" for n in G.nodes()]
    sizes = [260 if n == 0 else 80 for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color="#cccccc", width=0.4)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=axes[0],
        node_color=colors,
        node_size=sizes,
        edgecolors="black",
        linewidths=0.5,
    )
    axes[0].set_axis_off()
    axes[0].set_title("Star (1 hub)\nHub = incoherent signal")

    # line graph (chain) — gradient colored by position, ascending interaction count
    n_line = 50
    G = nx.path_graph(n_line)
    pos = {i: (i, 0) for i in range(n_line)}
    cmap = plt.get_cmap("YlOrRd")
    colors = [cmap(0.15 + 0.8 * (i / (n_line - 1))) for i in range(n_line)]
    sizes = [60 + 200 * (i / (n_line - 1)) for i in range(n_line)]
    nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color="#999999", width=0.6)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=axes[1],
        node_color=colors,
        node_size=sizes,
        edgecolors="black",
        linewidths=0.4,
    )
    axes[1].set_axis_off()
    axes[1].set_xlim(-2, n_line + 1)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].text(
        0,
        -0.7,
        "Sparse end\n(3 interactions)",
        fontsize=ANN_FONTSIZE,
        ha="center",
        va="top",
        color="#555",
    )
    axes[1].text(
        n_line - 1,
        -0.7,
        "Rich end\n(30 interactions)",
        fontsize=ANN_FONTSIZE,
        ha="center",
        va="top",
        color="#555",
    )
    axes[1].set_title("Line Graph (chain)\nPosition-dependent density")

    # fake users — ER base + injected bots
    n_real = 50
    n_fake = 6
    n_total = n_real + n_fake
    G = nx.erdos_renyi_graph(n_real, 6.0 / (n_real - 1), seed=SEED)
    rng_local = np.random.default_rng(SEED)
    for fake_id in range(n_real, n_total):
        G.add_node(fake_id)
        deg = int(rng_local.integers(2, 8))
        targets = rng_local.choice(n_real, size=deg, replace=False)
        for t in targets:
            G.add_edge(fake_id, int(t))
    pos = nx.spring_layout(G, seed=SEED)
    colors = ["#d62728" if n >= n_real else "#4C72B0" for n in G.nodes()]
    sizes = [180 if n >= n_real else 100 for n in G.nodes()]
    real_edges = [(u, v) for u, v in G.edges() if u < n_real and v < n_real]
    bot_edges = [(u, v) for u, v in G.edges() if u >= n_real or v >= n_real]
    nx.draw_networkx_edges(G, pos, ax=axes[2], edgelist=real_edges, edge_color="#cccccc", width=0.5)
    nx.draw_networkx_edges(G, pos, ax=axes[2], edgelist=bot_edges, edge_color="#d62728", width=1.1)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=axes[2],
        node_color=colors,
        node_size=sizes,
        edgecolors="black",
        linewidths=0.5,
    )
    axes[2].set_axis_off()
    axes[2].set_title(r"Fake Users ($\phi = 0.10$)" + "\nBots (red) in ER base")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "group3_topologies.png"))
    plt.close(fig)
    print("wrote group3_topologies.png")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    group1()
    group2()
    group3()
