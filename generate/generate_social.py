"""Generate synthetic datasets for echo-chamber, partial-alignment, and contrarian topologies."""

import argparse
import os
import random

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

SEED = 1  # overridden by --seed at runtime


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_item_interests(n_items: int, n_dim: int = 10) -> np.ndarray:
    vecs = np.random.randn(n_items, n_dim)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def make_er_graph(n_users: int, avg_degree: float = 6.0) -> nx.Graph:
    """Erdős-Rényi graph calibrated to a target average degree (undirected for community/2-coloring)."""
    p = avg_degree / (n_users - 1)
    G = nx.erdos_renyi_graph(n_users, p, seed=SEED)
    print(f"  ER graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    degrees = [d for _, d in G.degree()]
    print(f"  Avg degree: {np.mean(degrees):.2f}  Max: {max(degrees)}  Min: {min(degrees)}")
    return G


def detect_communities(G: nx.Graph) -> list[set]:
    """Greedy modularity community detection (Clauset-Newman-Moore)."""
    communities = list(greedy_modularity_communities(G))
    sizes = sorted([len(c) for c in communities], reverse=True)
    print(f"  Detected {len(communities)} communities")
    print(f"  Sizes (top 10): {sizes[:10]}")
    return communities


def community_assignment(n_users: int, communities: list[set]) -> np.ndarray:
    assignment = np.full(n_users, -1, dtype=int)
    for c_idx, community in enumerate(communities):
        for user in community:
            assignment[user] = c_idx
    # isolated nodes become singleton communities
    next_c = len(communities)
    for u in range(n_users):
        if assignment[u] == -1:
            assignment[u] = next_c
            next_c += 1
    return assignment


def greedy_two_color(G: nx.Graph, n_users: int) -> np.ndarray:
    """BFS 2-coloring (ER graphs are not bipartite, so residual within-group edges are reported as impurity)."""
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

    color[color == -1] = 0  # isolated nodes → group A

    cross = sum(1 for u, v in G.edges() if color[u] != color[v])
    total = G.number_of_edges()
    g = [(color == 0).sum(), (color == 1).sum()]
    print(f"  Group sizes: {g[0]} (A) vs {g[1]} (B)")
    print(f"  Cross-group edges (anti-correlated): {cross}/{total} ({100 * cross / total:.1f}%)")
    print(
        f"  Within-group edges (impurity):       {total - cross}/{total} ({100 * (total - cross) / total:.1f}%)"
    )

    return color


def interests_echo_chamber(
    n_users: int,
    n_dim: int,
    communities: list[set],
) -> np.ndarray:
    """Users in the same community share an identical interest vector."""
    assignment = community_assignment(n_users, communities)
    n_communities = int(assignment.max()) + 1
    community_vecs = np.array([unit(np.random.randn(n_dim)) for _ in range(n_communities)])
    return np.array([community_vecs[assignment[u]] for u in range(n_users)])


def interests_partial_alignment(
    n_users: int,
    n_dim: int,
    communities: list[set],
    shared_dims: int = 5,
) -> np.ndarray:
    """Users in the same community share `shared_dims` interest dims; the rest are personal."""
    assert 1 <= shared_dims < n_dim, "shared_dims must be between 1 and n_dim-1"
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


def interests_contrarian(
    n_users: int,
    n_dim: int,
    color: np.ndarray,
) -> np.ndarray:
    """Group A gets +v, group B gets -v (plus small noise), producing anti-correlated neighbours."""
    base_vec = unit(np.random.randn(n_dim))
    vecs = np.zeros((n_users, n_dim))
    for u in range(n_users):
        noise = 0.05 * np.random.randn(n_dim)
        if color[u] == 0:
            vecs[u] = unit(base_vec + noise)
        else:
            vecs[u] = unit(-base_vec + noise)
    return vecs


def print_interest_diagnostics(
    G: nx.Graph,
    user_vecs: np.ndarray,
    sample_size: int = 500,
) -> None:
    """Compare cosine similarity of connected pairs versus random pairs."""
    edges = list(G.edges())
    sampled_edges = random.sample(edges, min(sample_size, len(edges)))
    edge_sims = [float(user_vecs[u] @ user_vecs[v]) for u, v in sampled_edges]

    n = len(user_vecs)
    random_pairs = [
        (random.randint(0, n - 1), random.randint(0, n - 1)) for _ in range(sample_size)
    ]
    random_sims = [float(user_vecs[u] @ user_vecs[v]) for u, v in random_pairs if u != v]

    print(
        f"  Connected users:  mean={np.mean(edge_sims):+.3f}  "
        f"std={np.std(edge_sims):.3f}  "
        f"min={np.min(edge_sims):+.3f}  max={np.max(edge_sims):+.3f}"
    )
    print(f"  Random pairs:     mean={np.mean(random_sims):+.3f}  std={np.std(random_sims):.3f}")
    print(f"  Signal gap (connected - random): {np.mean(edge_sims) - np.mean(random_sims):+.3f}")


def cosine_to_rating(sim: float) -> float:
    """Maps cosine similarity [-1, 1] to a rating in [1, 5] with Gaussian noise."""
    raw = 1.0 + 2.0 * (sim + 1.0)
    return float(np.clip(round(raw + np.random.normal(0, 0.5)), 1, 5))


def generate_interactions(
    n_users: int,
    n_items: int,
    user_vecs: np.ndarray,
    item_vecs: np.ndarray,
    sparsity: float = 0.0005,
    noise: float = 0.20,
) -> list[tuple[int, int, float]]:
    """Per-user interaction counts are log-normal around the sparsity-implied mean."""
    mean_interactions = max(1, int(sparsity * n_users * n_items) // n_users)
    # cap prevents RecBole negative-sampling failure
    max_interactions = int(n_items * 0.1)

    log_mean = np.log(mean_interactions)
    log_std = 0.5
    counts = np.random.lognormal(log_mean, log_std, n_users).astype(int)
    counts = np.clip(counts, 1, max_interactions)

    pairs: dict[tuple[int, int], float] = {}

    chunk = 500
    for u_start in range(0, n_users, chunk):
        u_end = min(u_start + chunk, n_users)
        sims = user_vecs[u_start:u_end] @ item_vecs.T
        for i, u in enumerate(range(u_start, u_end)):
            per_user = int(counts[u])
            probs = softmax(sims[i])
            n_clean = int(per_user * (1 - noise))
            n_noisy = per_user - n_clean
            chosen = np.random.choice(n_items, size=n_clean, replace=False, p=probs)
            noisy = np.random.choice(n_items, size=n_noisy, replace=False)
            for item in np.concatenate([chosen, noisy]):
                item = int(item)
                if (u, item) not in pairs:
                    pairs[(u, item)] = cosine_to_rating(float(sims[i, item]))

    return [(u, item, r) for (u, item), r in pairs.items()]


def write_inter(path: str, interactions: list[tuple[int, int, float]]) -> None:
    with open(path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for user_id, item_id, rating in interactions:
            f.write(f"{user_id}\t{item_id}\t{rating}\n")
    print(f"  Wrote {len(interactions):,} interactions → {path}")


def write_net(path: str, G: nx.Graph) -> None:
    edges = list(G.edges())
    with open(path, "w") as f:
        f.write("source_id:token\ttarget_id:token\n")
        for src, tgt in edges:
            f.write(f"{src}\t{tgt}\n")
            f.write(f"{tgt}\t{src}\n")
    print(f"  Wrote {len(edges) * 2:,} directed social edges → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RecBole datasets (interest-vector topologies, fixed ER graph)"
    )
    parser.add_argument(
        "--topology", choices=["echo_chamber", "partial_alignment", "contrarian"], required=True
    )
    parser.add_argument("--n_users", type=int, default=2000)
    parser.add_argument("--n_items", type=int, default=10000)
    parser.add_argument("--n_dim", type=int, default=10)
    parser.add_argument(
        "--avg_degree",
        type=float,
        default=6.0,
        help="Target average degree for ER graph (same for all three)",
    )
    parser.add_argument(
        "--shared_dims",
        type=int,
        default=5,
        help="Partial alignment only: shared dimensions within community",
    )
    parser.add_argument("--sparsity", type=float, default=0.0005)
    parser.add_argument("--noise", type=float, default=0.20)
    parser.add_argument("--out_dir", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset generation")
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)

    dataset_name = (
        f"{args.topology}_u{args.n_users}_i{args.n_items}_sp{args.sparsity}_ns{args.noise}"
    )

    if args.topology == "partial_alignment" and args.shared_dims != 5:
        dataset_name += f"_sd{args.shared_dims}"
    if args.avg_degree != 6.0:
        dataset_name += f"_deg{args.avg_degree}"

    out_dir = os.path.join(args.out_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, f"{dataset_name}.inter")):
        print(f"⏭️  Skipping {dataset_name} — already exists in {out_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Topology     : {args.topology}")
    print(f"  Users        : {args.n_users:,}   Items: {args.n_items:,}")
    print(f"  Avg degree   : {args.avg_degree}  (ER graph, same for all three)")
    print(f"  Sparsity     : {args.sparsity}   Noise: {args.noise}")
    print(f"  Output       : {out_dir}")
    print(f"{'=' * 60}\n")

    print("[1/4] Building ER random graph...")
    G = make_er_graph(args.n_users, avg_degree=args.avg_degree)

    print("[2/4] Analyzing graph structure...")
    if args.topology in ("echo_chamber", "partial_alignment"):
        communities = detect_communities(G)
    elif args.topology == "contrarian":
        color = greedy_two_color(G, args.n_users)

    print("[3/4] Assigning interest vectors...")
    item_vecs = make_item_interests(args.n_items, args.n_dim)

    if args.topology == "echo_chamber":
        user_vecs = interests_echo_chamber(args.n_users, args.n_dim, communities)
    elif args.topology == "partial_alignment":
        user_vecs = interests_partial_alignment(
            args.n_users, args.n_dim, communities, shared_dims=args.shared_dims
        )
    elif args.topology == "contrarian":
        user_vecs = interests_contrarian(args.n_users, args.n_dim, color)

    print_interest_diagnostics(G, user_vecs)

    print("[4/4] Generating interactions and writing RecBole files...")
    interactions = generate_interactions(
        args.n_users,
        args.n_items,
        user_vecs,
        item_vecs,
        sparsity=args.sparsity,
        noise=args.noise,
    )
    actual_sparsity = len(interactions) / (args.n_users * args.n_items)
    print(f"  Target sparsity: {args.sparsity:.4%}  Actual: {actual_sparsity:.4%}")

    write_inter(os.path.join(out_dir, f"{dataset_name}.inter"), interactions)
    write_net(os.path.join(out_dir, f"{dataset_name}.net"), G)

    print("\nDone! RecBole config:")
    print(f"  dataset: '{dataset_name}'")
    print(f"  data_path: '{args.out_dir}'\n")


if __name__ == "__main__":
    main()
