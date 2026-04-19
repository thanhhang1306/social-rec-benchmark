"""Statistics for §4.7.1: paired-bootstrap tuning (aggregate and per-topology tables)."""

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
N_RESAMPLES = 5000
R = Path("/n/fs/recbench/new_rec/results")

PRIMARY_TOPOS = [
    ("Random", "random_u2000_i5000_sp0.01_ns0.1"),
    ("Echo Chamber", "echo_chamber_u2000_i5000_sp0.01_ns0.1"),
    ("Partial Align.", "partial_alignment_u2000_i5000_sp0.01_ns0.1"),
    ("Contrarian", "contrarian_u2000_i5000_sp0.01_ns0.1"),
    ("Scale-Free", "scale_free_u2000_i5000_sp0.01_ns0.1"),
    ("Small-World", "small_world_u2000_i5000_sp0.01_ns0.1"),
    ("Star", "star_u2000_i5000_sp0.01_ns0.1"),
    ("Line Graph", "line_graph_u2000_i5000_sp0.01_ns0.1_max30_decay0.5"),
    ("Fake Users", "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.1"),
]
MODELS = ["BPR", "LightGCN", "DiffNet", "MHCN", "SimGCL"]
DSEEDS = [1, 2, 3]
MSEEDS = [1, 12, 123]


def load(batch, d, s, topo, m):
    p = R / batch / f"dseed_{d}" / f"mseed_{s}" / topo / m / "final_metrics.csv"
    return float(pd.read_csv(p)["ndcg@10"].iloc[0]) if p.exists() else None


def paired_bootstrap_ci(diffs, rng):
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rng = np.random.default_rng(RNG_SEED)

    # per-topology × per-model diffs (9 obs each)
    per_topo = {}
    for tn, td in PRIMARY_TOPOS:
        for m in MODELS:
            diffs = []
            for d in DSEEDS:
                for s in MSEEDS:
                    t = load("TUNED_BATCH_SYNTH", d, s, td, m)
                    u = load("DEFAULT_BATCH_SYNTH", d, s, td, m)
                    if t is None or u is None:
                        continue
                    diffs.append(t - u)
            mean = float(np.mean(diffs))
            lo, hi = paired_bootstrap_ci(diffs, rng)
            per_topo[(tn, m)] = (mean, lo, hi, len(diffs))

    # aggregate per model (81 obs each)
    agg = {}
    for m in MODELS:
        diffs = []
        for tn, td in PRIMARY_TOPOS:
            for d in DSEEDS:
                for s in MSEEDS:
                    t = load("TUNED_BATCH_SYNTH", d, s, td, m)
                    u = load("DEFAULT_BATCH_SYNTH", d, s, td, m)
                    if t is None or u is None:
                        continue
                    diffs.append(t - u)
        mean = float(np.mean(diffs))
        lo, hi = paired_bootstrap_ci(diffs, rng)
        sig_plus = sum(1 for (tn_, m_) in per_topo if m_ == m and per_topo[(tn_, m_)][1] > 0)
        agg[m] = (mean, lo, hi, len(diffs), sig_plus)

    print("Table 4.7 aggregate (n=81, 5000 bootstrap, seed=42)")
    print(f"{'Model':<9} {'mean':>9} {'CI lo':>10} {'CI hi':>10}  Sig.+/9")
    for m in MODELS:
        mean, lo, hi, n, sig = agg[m]
        print(f"{m:<9} {mean:+.4f}   [{lo:+.4f}, {hi:+.4f}]    {sig}/9")

    print("\nTable C.2 per-topology (n=9, 5000 bootstrap, seed=42)")
    for tn, _ in PRIMARY_TOPOS:
        for m in MODELS:
            mean, lo, hi, n = per_topo[(tn, m)]
            star = "*" if lo > 0 else " "
            print(f"{tn:<16} {m:<9} {mean:+.4f}   [{lo:+.4f}, {hi:+.4f}] {star}")
        print()

    return agg, per_topo


if __name__ == "__main__":
    main()
