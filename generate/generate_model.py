"""Generate synthetic datasets for scale-free, small-world, and random topologies."""

import argparse
import os
import random

import networkx as nx
import numpy as np

SEED = 1  # overridden by --seed at runtime


def make_user_interests(n_users: int, n_dim: int = 10) -> np.ndarray:
    vecs = np.random.randn(n_users, n_dim)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_item_interests(n_items: int, n_dim: int = 10) -> np.ndarray:
    vecs = np.random.randn(n_items, n_dim)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def make_scale_free(n_users: int, ba_m: int = 3) -> nx.DiGraph:
    """Barabási-Albert preferential attachment (ba_m edges per new node)."""
    G_undirected = nx.barabasi_albert_graph(n_users, ba_m, seed=SEED)
    return G_undirected.to_directed()


def make_small_world(n_users: int, k: int = 6, p: float = 0.1) -> nx.DiGraph:
    """Watts-Strogatz (k ring neighbours, rewiring probability p)."""
    G_undirected = nx.watts_strogatz_graph(n_users, k, p, seed=SEED)
    return G_undirected.to_directed()


def make_random(n_users: int, p: float = 0.003) -> nx.DiGraph:
    """Erdős-Rényi (edge probability p)."""
    G_undirected = nx.erdos_renyi_graph(n_users, p, seed=SEED)
    return G_undirected.to_directed()


TOPOLOGY_BUILDERS = {
    "scale_free": make_scale_free,
    "small_world": make_small_world,
    "random": make_random,
}


def cosine_to_rating(sim: float) -> float:
    """Map cosine similarity in [-1, 1] to an integer rating in [1, 5] with Gaussian noise."""
    raw = 1.0 + 2.0 * (sim + 1.0)
    noisy = raw + np.random.normal(0, 0.5)
    return float(np.clip(round(noisy), 1, 5))


def generate_interactions(
    n_users: int,
    n_items: int,
    user_vecs: np.ndarray,
    item_vecs: np.ndarray,
    sparsity: float = 0.0005,
    noise: float = 0.20,
) -> list[tuple[int, int, float]]:
    """Sample items per user proportional to cosine similarity, then randomise `noise` fraction."""
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
        sims = user_vecs[u_start:u_end] @ item_vecs.T  # (chunk, n_items)

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
                    rating = cosine_to_rating(float(sims[i, item]))
                    pairs[(u, item)] = rating

    return [(u, item, r) for (u, item), r in pairs.items()]


def write_inter(path: str, interactions: list[tuple[int, int, float]]) -> None:
    with open(path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for user_id, item_id, rating in interactions:
            f.write(f"{user_id}\t{item_id}\t{rating}\n")
    print(f"  Wrote {len(interactions):,} interactions → {path}")


def write_net(path: str, G: nx.DiGraph) -> None:
    edges = list(G.edges())
    with open(path, "w") as f:
        f.write("source_id:token\ttarget_id:token\n")
        for src, tgt in edges:
            f.write(f"{src}\t{tgt}\n")
    print(f"  Wrote {len(edges):,} social edges → {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RecBole datasets")
    parser.add_argument("--topology", choices=list(TOPOLOGY_BUILDERS.keys()), required=True)
    parser.add_argument("--n_users", type=int, default=2000)
    parser.add_argument("--n_items", type=int, default=10000)
    parser.add_argument("--n_dim", type=int, default=10, help="Interest vector dimensionality")
    parser.add_argument(
        "--sparsity", type=float, default=0.002, help="Fraction of user-item matrix to fill"
    )
    parser.add_argument(
        "--noise", type=float, default=0.20, help="Fraction of interactions that are random noise"
    )
    parser.add_argument("--out_dir", type=str, default="./datasets")
    parser.add_argument("--ba_m", type=int, default=3, help="BA model: edges per new node")
    parser.add_argument("--ws_k", type=int, default=6, help="WS model: k nearest neighbours")
    parser.add_argument("--ws_p", type=float, default=0.1, help="WS model: rewiring probability")
    parser.add_argument(
        "--er_p", type=float, default=None, help="ER model: edge probability (auto if None)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset generation")
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)

    # dataset name encodes its key properties
    dataset_name = (
        f"{args.topology}_u{args.n_users}_i{args.n_items}_sp{args.sparsity}_ns{args.noise}"
    )

    if args.topology == "scale_free" and args.ba_m != 3:
        dataset_name += f"_bam{args.ba_m}"
    if args.topology == "small_world" and args.ws_k != 6:
        dataset_name += f"_wsk{args.ws_k}"
    if args.topology == "random" and args.er_p is not None:
        dataset_name += f"_erp{args.er_p}"

    out_dir = os.path.join(args.out_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, f"{dataset_name}.inter")):
        print(f"⏭️  Skipping {dataset_name} — already exists in {out_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Topology : {args.topology}")
    print(f"  Users    : {args.n_users:,}   Items: {args.n_items:,}")
    print(f"  Sparsity : {args.sparsity}   Noise: {args.noise}")
    print(f"  Output   : {out_dir}")
    print(f"{'=' * 60}\n")

    print("[1/4] Building social graph...")
    builder = TOPOLOGY_BUILDERS[args.topology]
    if args.topology == "scale_free":
        G = builder(args.n_users, ba_m=args.ba_m)
    elif args.topology == "small_world":
        G = builder(args.n_users, k=args.ws_k, p=args.ws_p)
    elif args.topology == "random":
        # auto-calibrate ER p to match average degree of BA/WS (~2*ba_m or ws_k)
        if args.er_p is None:
            target_avg_degree = 6  # matches default WS k=6
            auto_p = target_avg_degree / (args.n_users - 1)
            print(f"  Auto ER p = {auto_p:.5f} (targets avg degree ≈ {target_avg_degree})")
        else:
            auto_p = args.er_p
        G = builder(args.n_users, p=auto_p)

    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    print(f"  Avg degree: {np.mean([d for _, d in G.degree()]):.2f}")

    print("[2/4] Generating interest vectors...")
    user_vecs = make_user_interests(args.n_users, args.n_dim)
    item_vecs = make_item_interests(args.n_items, args.n_dim)

    print("[3/4] Sampling interactions...")
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

    print("[4/4] Writing RecBole files...")
    write_inter(os.path.join(out_dir, f"{dataset_name}.inter"), interactions)
    write_net(os.path.join(out_dir, f"{dataset_name}.net"), G)

    print("\nDone! RecBole config:")
    print(f"  dataset: '{dataset_name}'")
    print(f"  data_path: '{args.out_dir}'\n")


if __name__ == "__main__":
    main()
