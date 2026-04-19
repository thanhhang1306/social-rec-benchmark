"""Regenerate interest vectors and save cosine/Jaccard diagnostics per synthetic dataset."""

import argparse
import json
import os
import random as stdlib_random
import re

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities

def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_user_interests(n_users: int, n_dim: int = 10) -> np.ndarray:
    vecs = np.random.randn(n_users, n_dim)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def make_item_interests(n_items: int, n_dim: int = 10) -> np.ndarray:
    vecs = np.random.randn(n_items, n_dim)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def community_assignment(n_users: int, communities: list) -> np.ndarray:
    assignment = np.full(n_users, -1, dtype=int)
    for c_idx, community in enumerate(communities):
        for user in community:
            assignment[user] = c_idx
    next_c = len(communities)
    for u in range(n_users):
        if assignment[u] == -1:
            assignment[u] = next_c
            next_c += 1
    return assignment


def interests_echo_chamber(n_users, n_dim, communities):
    assignment = community_assignment(n_users, communities)
    n_communities = int(assignment.max()) + 1
    community_vecs = np.array([unit(np.random.randn(n_dim)) for _ in range(n_communities)])
    return np.array([community_vecs[assignment[u]] for u in range(n_users)])


def interests_partial_alignment(n_users, n_dim, communities, shared_dims=5):
    assignment = community_assignment(n_users, communities)
    n_communities = int(assignment.max()) + 1
    community_shared = np.random.randn(n_communities, shared_dims)
    vecs = np.zeros((n_users, n_dim))
    for u in range(n_users):
        c = assignment[u]
        shared_part = community_shared[c]
        personal_part = np.random.randn(n_dim - shared_dims)
        vecs[u] = unit(np.concatenate([shared_part, personal_part]))
    return vecs


def interests_contrarian(n_users, n_dim, color):
    base_vec = unit(np.random.randn(n_dim))
    vecs = np.zeros((n_users, n_dim))
    for u in range(n_users):
        noise = 0.05 * np.random.randn(n_dim)
        if color[u] == 0:
            vecs[u] = unit(base_vec + noise)
        else:
            vecs[u] = unit(-base_vec + noise)
    return vecs


def greedy_two_color(G, n_users):
    color = np.full(n_users, -1, dtype=int)
    for start in range(n_users):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbour in G.neighbors(node):
                if color[neighbour] == -1:
                    color[neighbour] = 1 - color[node]
                    queue.append(neighbour)
    color[color == -1] = 0
    return color


def compute_cosine_diagnostics(G, user_vecs, sample_size=500):
    """Cosine similarity between connected users vs random pairs."""
    edges = list(G.edges())
    if not edges:
        return {}

    sampled_edges = stdlib_random.sample(edges, min(sample_size, len(edges)))
    edge_sims = [float(user_vecs[u] @ user_vecs[v]) for u, v in sampled_edges]

    n = len(user_vecs)
    random_pairs = [
        (stdlib_random.randint(0, n - 1), stdlib_random.randint(0, n - 1))
        for _ in range(sample_size)
    ]
    random_sims = [float(user_vecs[u] @ user_vecs[v]) for u, v in random_pairs if u != v]

    return {
        "connected_cosine_mean": float(np.mean(edge_sims)),
        "connected_cosine_std": float(np.std(edge_sims)),
        "connected_cosine_min": float(np.min(edge_sims)),
        "connected_cosine_max": float(np.max(edge_sims)),
        "random_cosine_mean": float(np.mean(random_sims)),
        "random_cosine_std": float(np.std(random_sims)),
        "signal_gap": float(np.mean(edge_sims) - np.mean(random_sims)),
    }


def compute_jaccard_diagnostics(G, inter_df, sample_size=500):
    """Empirical Jaccard similarity of interaction sets (from .inter + .net)."""
    user_items = inter_df.groupby("user_id")["item_id"].apply(set).to_dict()

    edges = [(u, v) for u, v in G.edges() if u in user_items and v in user_items]
    if not edges:
        return {}

    sampled_edges = stdlib_random.sample(edges, min(sample_size, len(edges)))
    edge_jaccards = []
    for u, v in sampled_edges:
        su, sv = user_items[u], user_items[v]
        inter = len(su & sv)
        union = len(su | sv)
        edge_jaccards.append(inter / union if union > 0 else 0.0)

    users = list(user_items.keys())
    random_jaccards = []
    for _ in range(sample_size):
        u, v = stdlib_random.choice(users), stdlib_random.choice(users)
        if u == v:
            continue
        su, sv = user_items[u], user_items[v]
        inter = len(su & sv)
        union = len(su | sv)
        random_jaccards.append(inter / union if union > 0 else 0.0)

    return {
        "connected_jaccard_mean": float(np.mean(edge_jaccards)),
        "connected_jaccard_std": float(np.std(edge_jaccards)),
        "random_jaccard_mean": float(np.mean(random_jaccards)),
        "random_jaccard_std": float(np.std(random_jaccards)),
        "jaccard_signal_gap": float(np.mean(edge_jaccards) - np.mean(random_jaccards)),
    }


SOCIAL_TOPOS = {"echo_chamber", "partial_alignment", "contrarian"}
MODEL_TOPOS = {"random", "scale_free", "small_world"}
STRUCT_TOPOS = {"star", "line_graph", "fake_users"}


def parse_dataset_name(name):
    """Extract topology and generation params from dataset folder name."""
    # identify topology (longest match first)
    topology = None
    for t in sorted(SOCIAL_TOPOS | MODEL_TOPOS | STRUCT_TOPOS, key=len, reverse=True):
        if name.startswith(t + "_"):
            topology = t
            break
    if topology is None:
        return None

    params = {"topology": topology}

    m = re.search(r"_u(\d+)", name)
    if m:
        params["n_users"] = int(m.group(1))
    m = re.search(r"_i(\d+)", name)
    if m:
        params["n_items"] = int(m.group(1))
    m = re.search(r"_sp([\d.]+)", name)
    if m:
        params["sparsity"] = float(m.group(1))
    m = re.search(r"_ns([\d.]+)", name)
    if m:
        params["noise"] = float(m.group(1))
    m = re.search(r"_sd(\d+)", name)
    if m:
        params["shared_dims"] = int(m.group(1))
    m = re.search(r"_deg([\d.]+)", name)
    if m:
        params["avg_degree"] = float(m.group(1))
    m = re.search(r"_fake([\d.]+)", name)
    if m:
        params["fake_ratio"] = float(m.group(1))
    m = re.search(r"_erp([\d.]+)", name)
    if m:
        params["er_p"] = float(m.group(1))
    m = re.search(r"_max(\d+)", name)
    if m:
        params["max_interactions"] = int(m.group(1))
    m = re.search(r"_decay([\d.]+)", name)
    if m:
        params["decay"] = float(m.group(1))

    return params


def load_graph_from_net(net_path, n_users=None):
    """Load a directed graph from a RecBole .net file; add isolated nodes up to n_users if given."""
    G = nx.DiGraph()
    if n_users:
        G.add_nodes_from(range(n_users))
    with open(net_path) as f:
        header = f.readline()  # skip header  # noqa: F841
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


def load_inter(inter_path):
    """Load .inter file as DataFrame."""
    df = pd.read_csv(inter_path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    return df


def regenerate_user_vecs(params, seed, G):
    """Replay the generator's RNG call sequence to reproduce user interest vectors."""
    topo = params["topology"]
    n_users = params.get("n_users", 2000)
    n_items = params.get("n_items", 10000)
    n_dim = 10
    meta = {}

    np.random.seed(seed)
    stdlib_random.seed(seed)

    if topo in MODEL_TOPOS:
        # generate_model.py: nx uses seed= param (not numpy)
        # first numpy call is user_vecs
        user_vecs = make_user_interests(n_users, n_dim)

    elif topo == "star":
        # generate_topology.py StarTopology: nx.star_graph is deterministic
        # first numpy call is user_vecs, then hub override
        user_vecs = make_user_interests(n_users, n_dim)
        user_vecs[0] = unit(np.random.randn(n_dim))

    elif topo == "line_graph":
        # generate_topology.py LineGraphTopology: nx.path_graph is deterministic
        # first numpy call is user_vecs
        user_vecs = make_user_interests(n_users, n_dim)

    elif topo == "fake_users":
        # generate_topology.py FakeUsersTopology:
        # nx.erdos_renyi_graph uses seed= (not numpy)
        # then numpy used for fake connections before user_vecs
        fake_ratio = params.get("fake_ratio", 0.1)
        n_fake = int(n_users * fake_ratio)
        n_real = n_users - n_fake

        # replay the fake connection loop to advance numpy state
        for _ in range(n_fake):
            fake_degree = int(np.random.randint(1, 21))
            np.random.choice(n_real, size=min(fake_degree, n_real), replace=False)

        real_vecs = make_user_interests(n_real, n_dim)
        fake_vecs = make_user_interests(n_fake, n_dim)
        user_vecs = np.vstack([real_vecs, fake_vecs])

    elif topo in SOCIAL_TOPOS:
        # generate_social.py: nx.erdos_renyi_graph uses seed= (not numpy)
        # first numpy call is item_vecs (must advance past it)
        _ = make_item_interests(n_items, n_dim)  # advance RNG

        # community detection / coloring on loaded graph
        G_undirected = G.to_undirected()
        if topo in ("echo_chamber", "partial_alignment"):
            communities = list(greedy_modularity_communities(G_undirected))
            meta["n_communities"] = len(communities)
            meta["community_sizes_top5"] = sorted([len(c) for c in communities], reverse=True)[:5]
            if topo == "echo_chamber":
                user_vecs = interests_echo_chamber(n_users, n_dim, communities)
            else:
                sd = params.get("shared_dims", 5)
                user_vecs = interests_partial_alignment(n_users, n_dim, communities, sd)
        else:  # contrarian
            color = greedy_two_color(G_undirected, n_users)
            user_vecs = interests_contrarian(n_users, n_dim, color)

            # contrarian impurity: fraction of within-group edges (odd-cycle artefacts)
            total_edges = G_undirected.number_of_edges()
            if total_edges > 0:
                cross = sum(1 for u, v in G_undirected.edges() if color[u] != color[v])
                within = total_edges - cross
                meta["cross_group_edges"] = cross
                meta["within_group_edges"] = within
                meta["impurity_pct"] = 100.0 * within / total_edges
                meta["group_sizes"] = [int((color == 0).sum()), int((color == 1).sum())]

    else:
        return None, meta

    return user_vecs, meta


def process_one_dataset(ds_name, ds_path, seed):
    """Compute diagnostics for one dataset. Returns dict or None."""
    params = parse_dataset_name(ds_name)
    if params is None:
        print(f"  ⚠  Cannot parse: {ds_name}")
        return None

    net_path = os.path.join(ds_path, f"{ds_name}.net")
    inter_path = os.path.join(ds_path, f"{ds_name}.inter")

    if not os.path.exists(net_path) or not os.path.exists(inter_path):
        print(f"  ⚠  Missing files: {ds_name}")
        return None

    n_users = params.get("n_users", 2000)
    G = load_graph_from_net(net_path, n_users=n_users)
    inter_df = load_inter(inter_path)

    result = {
        "dataset": ds_name,
        "seed": seed,
        "topology": params["topology"],
    }

    # true cosine diagnostics (from regenerated vectors)
    user_vecs, meta = regenerate_user_vecs(params, seed, G)
    if user_vecs is not None:
        cosine = compute_cosine_diagnostics(G, user_vecs)
        result.update(cosine)
    else:
        print(f"  ⚠  Could not regenerate vectors for {ds_name}")

    # topology-specific metadata (contrarian impurity, community info, etc.)
    result.update(meta)

    # empirical Jaccard diagnostics (from .inter + .net)
    jaccard = compute_jaccard_diagnostics(G, inter_df)
    result.update(jaccard)

    diag_path = os.path.join(ds_path, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract cosine/Jaccard diagnostics for all synthetic datasets"
    )
    parser.add_argument(
        "--dseed", type=int, default=None, help="Single dseed to process (default: all)"
    )
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--out_csv", type=str, default="diagnostics_all.csv")
    args = parser.parse_args()

    if args.dseed is not None:
        dseeds = [args.dseed]
    else:
        dseeds = []
        for entry in sorted(os.listdir(args.datasets_dir)):
            m = re.match(r"dseed_(\d+)", entry)
            if m and os.path.isdir(os.path.join(args.datasets_dir, entry)):
                dseeds.append(int(m.group(1)))

    all_rows = []

    for dseed in sorted(dseeds):
        dseed_dir = os.path.join(args.datasets_dir, f"dseed_{dseed}")
        if not os.path.isdir(dseed_dir):
            print(f"⚠  {dseed_dir} not found, skipping")
            continue

        datasets = sorted(os.listdir(dseed_dir))
        print(f"\n{'=' * 60}")
        print(f"  dseed={dseed}  ({len(datasets)} datasets)")
        print(f"{'=' * 60}")

        for ds_name in datasets:
            ds_path = os.path.join(dseed_dir, ds_name)
            if not os.path.isdir(ds_path):
                continue

            row = process_one_dataset(ds_name, ds_path, dseed)
            if row is not None:
                all_rows.append(row)
                gap = row.get("signal_gap", float("nan"))
                jgap = row.get("jaccard_signal_gap", float("nan"))
                print(f"  ✅ {ds_name}  cosine_gap={gap:+.3f}  jaccard_gap={jgap:+.4f}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        front = ["dataset", "seed", "topology"]
        cols = front + [c for c in df.columns if c not in front]
        df = df[cols].sort_values(["seed", "topology", "dataset"]).reset_index(drop=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\n{'=' * 60}")
        print(f"Saved {len(df)} rows → {args.out_csv}")
        print("\nPer-topology signal gap summary (cosine):")
        print(df.groupby("topology")["signal_gap"].agg(["mean", "std", "min", "max"]).to_string())
        print("\nPer-topology signal gap summary (Jaccard):")
        print(
            df.groupby("topology")["jaccard_signal_gap"]
            .agg(["mean", "std", "min", "max"])
            .to_string()
        )
    else:
        print("No datasets processed.")


if __name__ == "__main__":
    main()
