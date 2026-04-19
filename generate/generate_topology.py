"""Generate synthetic datasets for star, line-graph, and fake-users topologies."""

import argparse
import os
import random

import networkx as nx
import numpy as np

SEED = 1  # overridden by --seed at runtime


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_user_interests(n_users: int, n_dim: int = 10) -> np.ndarray:
    vecs = np.random.randn(n_users, n_dim)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def make_item_interests(n_items: int, n_dim: int = 10) -> np.ndarray:
    vecs = np.random.randn(n_items, n_dim)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class StarTopology:
    """User 0 is a random-taste hub connected bidirectionally to every other user."""

    def build(
        self,
        n_users: int,
        n_items: int,
        n_dim: int,
        sparsity: float,
        noise: float,
    ) -> tuple[nx.DiGraph, np.ndarray, list[tuple[int, int]]]:
        # nx.star_graph(n) creates hub=node 0, spokes=nodes 1..n
        # returns undirected — convert to directed for bidirectional influence
        G = nx.star_graph(n_users - 1).to_directed()

        print(f"  Hub (user 0) degree: {G.degree(0)}  (= n_users - 1 = {n_users - 1})")

        user_vecs = make_user_interests(n_users, n_dim)
        # hub gets a fresh random vector — no relation to anyone else
        user_vecs[0] = unit(np.random.randn(n_dim))

        item_vecs = make_item_interests(n_items, n_dim)

        # hub: completely random items AND random ratings (no coherent taste)
        hub_n_interactions = max(1, int(sparsity * n_items))
        hub_interactions = [
            (0, int(i), float(np.random.randint(1, 6)))
            for i in np.random.choice(n_items, hub_n_interactions, replace=False)
        ]

        # spoke interactions: normal signal-based
        spoke_interactions = generate_interactions(
            n_users=n_users - 1,
            n_items=n_items,
            user_vecs=user_vecs[1:],  # users 1..n_users-1
            item_vecs=item_vecs,
            sparsity=sparsity,
            noise=noise,
            user_offset=1,  # shift IDs so user 1 stays user 1
        )

        # deduplicate by (user, item)
        seen: set[tuple[int, int]] = set()
        interactions = []
        for u, i, r in hub_interactions + spoke_interactions:
            if (u, i) not in seen:
                seen.add((u, i))
                interactions.append((u, i, r))
        return G, user_vecs, item_vecs, interactions


class LineGraphTopology:
    """Users in a bidirectional chain; interaction count decays with distance from the rich end.

    Requires RecBole filters user_inter_num_interval='[3, inf)' and item_inter_num_interval='[1, inf)'.
    """

    def build(
        self,
        n_users: int,
        n_items: int,
        n_dim: int,
        sparsity: float,
        noise: float,
        max_interactions: int = 30,
        min_interactions: int = 3,
        decay: float = 0.5,
    ) -> tuple[nx.DiGraph, np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
        G = nx.path_graph(n_users).to_directed()

        print(f"  Chain length: {n_users} users")
        print(
            f"  Interaction gradient: {min_interactions} (sparse end) → {max_interactions} (rich end)"
        )
        print(f"  Decay exponent: {decay}")
        print(
            "  ⚠  Use user_inter_num_interval: '[3, inf)' and item_inter_num_interval: '[1, inf)'"
        )

        user_vecs = make_user_interests(n_users, n_dim)
        item_vecs = make_item_interests(n_items, n_dim)

        pairs: dict[tuple[int, int], float] = {}

        for u in range(n_users):
            position_ratio = u / (n_users - 1)  # 0.0 (sparse) → 1.0 (rich)
            mean_count = max(min_interactions, int(max_interactions * (position_ratio**decay)))
            # small lognormal jitter around the gradient
            count = int(np.random.lognormal(np.log(mean_count), 0.3))
            count = int(np.clip(count, min_interactions, int(n_items * 0.1)))

            sims = user_vecs[u] @ item_vecs.T
            probs = softmax(sims)
            n_clean = int(count * (1 - noise))
            n_noisy = count - n_clean

            chosen = np.random.choice(n_items, size=n_clean, replace=False, p=probs)
            noisy = np.random.choice(n_items, size=n_noisy, replace=False)

            for item in np.concatenate([chosen, noisy]):
                item = int(item)
                if (u, item) not in pairs:
                    pairs[(u, item)] = cosine_to_rating(float(sims[item]))

        interactions = [(u, item, r) for (u, item), r in pairs.items()]

        counts_by_user = {}
        for u, _, _ in interactions:
            counts_by_user[u] = counts_by_user.get(u, 0) + 1
        all_counts = list(counts_by_user.values())
        print(
            f"  Interactions — min: {min(all_counts)}  "
            f"max: {max(all_counts)}  "
            f"mean: {np.mean(all_counts):.1f}  "
            f"total: {len(interactions):,}"
        )

        return G, user_vecs, item_vecs, interactions


class FakeUsersTopology:
    """ER base graph with `fake_ratio` bot users (random edges, random items, random vectors).

    Fake users occupy IDs n_real..n_users-1 so they can be isolated in post-analysis.
    """

    def build(
        self,
        n_users: int,
        n_items: int,
        n_dim: int,
        sparsity: float,
        noise: float,
        fake_ratio: float = 0.10,
    ) -> tuple[nx.DiGraph, np.ndarray, np.ndarray, list[tuple[int, int]]]:
        n_fake = int(n_users * fake_ratio)
        n_real = n_users - n_fake
        print(f"  Real users: {n_real}   Fake users: {n_fake} ({fake_ratio:.0%})")

        # base graph on real users (ER, calibrated to avg degree ~6)
        p_er = 6.0 / (n_real - 1)
        G_base = nx.erdos_renyi_graph(n_real, p_er, seed=SEED)
        G = G_base.to_directed()

        for fake_id in range(n_real, n_users):
            G.add_node(fake_id)

            fake_degree = np.random.randint(1, 21)

            targets = np.random.choice(n_real, size=min(fake_degree, n_real), replace=False)
            for t in targets:
                G.add_edge(fake_id, t)
                G.add_edge(t, fake_id)

        print(f"  Total edges: {G.number_of_edges():,}")

        real_vecs = make_user_interests(n_real, n_dim)
        fake_vecs = make_user_interests(n_fake, n_dim)
        user_vecs = np.vstack([real_vecs, fake_vecs])
        item_vecs = make_item_interests(n_items, n_dim)

        # real user interactions: signal-based
        real_interactions = generate_interactions(
            n_users=n_real,
            n_items=n_items,
            user_vecs=real_vecs,
            item_vecs=item_vecs,
            sparsity=sparsity,
            noise=noise,
            user_offset=0,
        )

        # fake user interactions: random items AND random ratings
        fake_interactions_per_user = max(1, int(sparsity * n_items))
        fake_interactions: list[tuple[int, int, float]] = []
        for fake_id in range(n_real, n_users):
            items = np.random.choice(n_items, size=fake_interactions_per_user, replace=False)
            for item in items:
                fake_interactions.append((fake_id, int(item), float(np.random.randint(1, 6))))

        # deduplicate by (user, item) — keep first seen
        seen: set[tuple[int, int]] = set()
        interactions = []
        for u, i, r in real_interactions + fake_interactions:
            if (u, i) not in seen:
                seen.add((u, i))
                interactions.append((u, i, r))
        print(f"  Real interactions: {len(real_interactions):,}   Fake: {len(fake_interactions):,}")
        return G, user_vecs, item_vecs, interactions


def cosine_to_rating(sim: float) -> float:
    """Map cosine similarity in [-1, 1] to an integer rating in [1, 5] with Gaussian noise."""
    raw = 1.0 + 2.0 * (sim + 1.0)
    return float(np.clip(round(raw + np.random.normal(0, 0.5)), 1, 5))


def generate_interactions(
    n_users: int,
    n_items: int,
    user_vecs: np.ndarray,
    item_vecs: np.ndarray,
    sparsity: float,
    noise: float,
    user_offset: int = 0,
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
                key = (u + user_offset, item)
                if key not in pairs:
                    pairs[key] = cosine_to_rating(float(sims[i, item]))

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


TOPOLOGY_MAP = {
    "star": StarTopology,
    "line_graph": LineGraphTopology,
    "fake_users": FakeUsersTopology,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RecBole datasets (structural stress tests)"
    )
    parser.add_argument("--topology", choices=list(TOPOLOGY_MAP.keys()), required=True)
    parser.add_argument("--n_users", type=int, default=2000)
    parser.add_argument("--n_items", type=int, default=10000)
    parser.add_argument("--n_dim", type=int, default=10)
    parser.add_argument("--sparsity", type=float, default=0.002)
    parser.add_argument("--noise", type=float, default=0.20)
    parser.add_argument(
        "--fake_ratio",
        type=float,
        default=0.10,
        help="Fraction of users that are bots (fake_users topology only)",
    )
    parser.add_argument(
        "--max_interactions",
        type=int,
        default=30,
        help="Interactions for richest user in chain (line_graph only)",
    )
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=3,
        help="Minimum interactions for sparsest user in chain (line_graph only)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.5,
        help="Decay exponent for interaction gradient (line_graph only)",
    )
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
    if args.topology == "fake_users":
        dataset_name += f"_fake{args.fake_ratio}"
    if args.topology == "line_graph":
        dataset_name += f"_max{args.max_interactions}_decay{args.decay}"

    out_dir = os.path.join(args.out_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, f"{dataset_name}.inter")):
        print(f"⏭️  Skipping {dataset_name} — already exists in {out_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Topology  : {args.topology}")
    print(f"  Users     : {args.n_users:,}   Items: {args.n_items:,}")
    print(f"  Sparsity  : {args.sparsity}   Noise: {args.noise}")
    print(f"  Output    : {out_dir}")
    print(f"{'=' * 60}\n")

    print("[1/3] Building graph + interest vectors...")
    topo = TOPOLOGY_MAP[args.topology]()

    if args.topology == "fake_users":
        G, user_vecs, item_vecs, interactions = topo.build(
            args.n_users,
            args.n_items,
            args.n_dim,
            args.sparsity,
            args.noise,
            fake_ratio=args.fake_ratio,
        )
    elif args.topology == "line_graph":
        G, user_vecs, item_vecs, interactions = topo.build(
            args.n_users,
            args.n_items,
            args.n_dim,
            args.sparsity,
            args.noise,
            max_interactions=args.max_interactions,
            min_interactions=args.min_interactions,
            decay=args.decay,
        )
    else:
        G, user_vecs, item_vecs, interactions = topo.build(
            args.n_users, args.n_items, args.n_dim, args.sparsity, args.noise
        )

    actual_sparsity = len(interactions) / (args.n_users * args.n_items)
    print(f"\n[2/3] Interactions: {len(interactions):,}  (actual sparsity: {actual_sparsity:.4%})")

    print("[3/3] Writing RecBole files...")
    write_inter(os.path.join(out_dir, f"{dataset_name}.inter"), interactions)
    write_net(os.path.join(out_dir, f"{dataset_name}.net"), G)

    print("\nDone! RecBole config:")
    print(f"  dataset: '{dataset_name}'")
    print(f"  data_path: '{args.out_dir}'\n")


if __name__ == "__main__":
    main()
