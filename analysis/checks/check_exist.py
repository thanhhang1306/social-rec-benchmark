"""Report done/missing/failed counts per batch."""

import os

RESULTS_DIR = "/n/fs/recbench/new_rec/results"

MODELS = ["BPR", "LightGCN", "SimGCL", "MHCN", "DiffNet", "HAGN", "CSGCN", "ASLGCN"]

REAL_DATASETS = ["yelp", "douban-book", "lastfm"]

SYNTH_DATASETS = [
    "contrarian_u2000_i10000_sp0.002_ns0.2",
    "echo_chamber_u2000_i10000_sp0.002_ns0.2",
    "fake_users_u2000_i10000_sp0.002_ns0.2_fake0.1",
    "line_graph_u2000_i10000_sp0.002_ns0.2_max30_decay0.5",
    "partial_alignment_u2000_i10000_sp0.002_ns0.2",
    "random_u2000_i10000_sp0.002_ns0.2",
    "scale_free_u2000_i10000_sp0.002_ns0.2",
    "small_world_u2000_i10000_sp0.002_ns0.2",
    "star_u2000_i10000_sp0.002_ns0.2",
    "contrarian_u2000_i5000_sp0.01_ns0.2",
    "echo_chamber_u2000_i5000_sp0.01_ns0.2",
    "fake_users_u2000_i5000_sp0.01_ns0.2_fake0.1",
    "line_graph_u2000_i5000_sp0.01_ns0.2_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.01_ns0.2",
    "random_u2000_i5000_sp0.01_ns0.2",
    "scale_free_u2000_i5000_sp0.01_ns0.2",
    "small_world_u2000_i5000_sp0.01_ns0.2",
    "star_u2000_i5000_sp0.01_ns0.2",
    "contrarian_u2000_i5000_sp0.01_ns0.1",
    "echo_chamber_u2000_i5000_sp0.01_ns0.1",
    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.1",
    "line_graph_u2000_i5000_sp0.01_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1",
    "random_u2000_i5000_sp0.01_ns0.1",
    "scale_free_u2000_i5000_sp0.01_ns0.1",
    "small_world_u2000_i5000_sp0.01_ns0.1",
    "star_u2000_i5000_sp0.01_ns0.1",
    "contrarian_u2000_i5000_sp0.02_ns0.1",
    "echo_chamber_u2000_i5000_sp0.02_ns0.1",
    "fake_users_u2000_i5000_sp0.02_ns0.1_fake0.1",
    "line_graph_u2000_i5000_sp0.02_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.02_ns0.1",
    "random_u2000_i5000_sp0.02_ns0.1",
    "scale_free_u2000_i5000_sp0.02_ns0.1",
    "small_world_u2000_i5000_sp0.02_ns0.1",
    "star_u2000_i5000_sp0.02_ns0.1",
    "contrarian_u2000_i5000_sp0.005_ns0.1",
    "echo_chamber_u2000_i5000_sp0.005_ns0.1",
    "fake_users_u2000_i5000_sp0.005_ns0.1_fake0.1",
    "line_graph_u2000_i5000_sp0.005_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.005_ns0.1",
    "random_u2000_i5000_sp0.005_ns0.1",
    "scale_free_u2000_i5000_sp0.005_ns0.1",
    "small_world_u2000_i5000_sp0.005_ns0.1",
    "star_u2000_i5000_sp0.005_ns0.1",
    "contrarian_u2000_i5000_sp0.02_ns0.2",
    "echo_chamber_u2000_i5000_sp0.02_ns0.2",
    "fake_users_u2000_i5000_sp0.02_ns0.2_fake0.1",
    "line_graph_u2000_i5000_sp0.02_ns0.2_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.02_ns0.2",
    "random_u2000_i5000_sp0.02_ns0.2",
    "scale_free_u2000_i5000_sp0.02_ns0.2",
    "small_world_u2000_i5000_sp0.02_ns0.2",
    "star_u2000_i5000_sp0.02_ns0.2",
    "contrarian_u2000_i5000_sp0.01_ns0.05",
    "echo_chamber_u2000_i5000_sp0.01_ns0.05",
    "fake_users_u2000_i5000_sp0.01_ns0.05_fake0.1",
    "line_graph_u2000_i5000_sp0.01_ns0.05_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.01_ns0.05",
    "random_u2000_i5000_sp0.01_ns0.05",
    "scale_free_u2000_i5000_sp0.01_ns0.05",
    "small_world_u2000_i5000_sp0.01_ns0.05",
    "star_u2000_i5000_sp0.01_ns0.05",
    "contrarian_u2000_i2000_sp0.01_ns0.1",
    "echo_chamber_u2000_i2000_sp0.01_ns0.1",
    "fake_users_u2000_i2000_sp0.01_ns0.1_fake0.1",
    "line_graph_u2000_i2000_sp0.01_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i2000_sp0.01_ns0.1",
    "random_u2000_i2000_sp0.01_ns0.1",
    "scale_free_u2000_i2000_sp0.01_ns0.1",
    "small_world_u2000_i2000_sp0.01_ns0.1",
    "star_u2000_i2000_sp0.01_ns0.1",
    "contrarian_u2000_i2000_sp0.02_ns0.1",
    "echo_chamber_u2000_i2000_sp0.02_ns0.1",
    "fake_users_u2000_i2000_sp0.02_ns0.1_fake0.1",
    "line_graph_u2000_i2000_sp0.02_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i2000_sp0.02_ns0.1",
    "random_u2000_i2000_sp0.02_ns0.1",
    "scale_free_u2000_i2000_sp0.02_ns0.1",
    "small_world_u2000_i2000_sp0.02_ns0.1",
    "star_u2000_i2000_sp0.02_ns0.1",
    "contrarian_u2000_i10000_sp0.01_ns0.1",
    "echo_chamber_u2000_i10000_sp0.01_ns0.1",
    "fake_users_u2000_i10000_sp0.01_ns0.1_fake0.1",
    "line_graph_u2000_i10000_sp0.01_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i10000_sp0.01_ns0.1",
    "random_u2000_i10000_sp0.01_ns0.1",
    "scale_free_u2000_i10000_sp0.01_ns0.1",
    "small_world_u2000_i10000_sp0.01_ns0.1",
    "star_u2000_i10000_sp0.01_ns0.1",
    "contrarian_u2000_i5000_sp0.01_ns0.3",
    "echo_chamber_u2000_i5000_sp0.01_ns0.3",
    "fake_users_u2000_i5000_sp0.01_ns0.3_fake0.1",
    "line_graph_u2000_i5000_sp0.01_ns0.3_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.01_ns0.3",
    "random_u2000_i5000_sp0.01_ns0.3",
    "scale_free_u2000_i5000_sp0.01_ns0.3",
    "small_world_u2000_i5000_sp0.01_ns0.3",
    "star_u2000_i5000_sp0.01_ns0.3",
    "contrarian_u2000_i5000_sp0.01_ns0.5",
    "echo_chamber_u2000_i5000_sp0.01_ns0.5",
    "fake_users_u2000_i5000_sp0.01_ns0.5_fake0.1",
    "line_graph_u2000_i5000_sp0.01_ns0.5_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.01_ns0.5",
    "random_u2000_i5000_sp0.01_ns0.5",
    "scale_free_u2000_i5000_sp0.01_ns0.5",
    "small_world_u2000_i5000_sp0.01_ns0.5",
    "star_u2000_i5000_sp0.01_ns0.5",
    "contrarian_u2000_i5000_sp0.01_ns0.02",
    "echo_chamber_u2000_i5000_sp0.01_ns0.02",
    "fake_users_u2000_i5000_sp0.01_ns0.02_fake0.1",
    "line_graph_u2000_i5000_sp0.01_ns0.02_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.01_ns0.02",
    "random_u2000_i5000_sp0.01_ns0.02",
    "scale_free_u2000_i5000_sp0.01_ns0.02",
    "small_world_u2000_i5000_sp0.01_ns0.02",
    "star_u2000_i5000_sp0.01_ns0.02",
    "contrarian_u2000_i5000_sp0.003_ns0.1",
    "echo_chamber_u2000_i5000_sp0.003_ns0.1",
    "fake_users_u2000_i5000_sp0.003_ns0.1_fake0.1",
    "line_graph_u2000_i5000_sp0.003_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.003_ns0.1",
    "random_u2000_i5000_sp0.003_ns0.1",
    "scale_free_u2000_i5000_sp0.003_ns0.1",
    "small_world_u2000_i5000_sp0.003_ns0.1",
    "star_u2000_i5000_sp0.003_ns0.1",
    "contrarian_u2000_i5000_sp0.05_ns0.1",
    "echo_chamber_u2000_i5000_sp0.05_ns0.1",
    "fake_users_u2000_i5000_sp0.05_ns0.1_fake0.1",
    "line_graph_u2000_i5000_sp0.05_ns0.1_max30_decay0.5",
    "partial_alignment_u2000_i5000_sp0.05_ns0.1",
    "random_u2000_i5000_sp0.05_ns0.1",
    "scale_free_u2000_i5000_sp0.05_ns0.1",
    "small_world_u2000_i5000_sp0.05_ns0.1",
    "star_u2000_i5000_sp0.05_ns0.1",
    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.05",
    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.2",
    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.4",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd1",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd3",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd7",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd9",
    "random_u2000_i5000_sp0.01_ns0.1_erp0.001",
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg2.0",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg2.0",
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg2.0",
    "random_u2000_i5000_sp0.01_ns0.1_erp0.002",
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg4.0",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg4.0",
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg4.0",
    "random_u2000_i5000_sp0.01_ns0.1_erp0.005",
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg10.0",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg10.0",
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg10.0",
    "random_u2000_i5000_sp0.01_ns0.1_erp0.01",
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg20.0",
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg20.0",
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg20.0",
    "contrarian_u500_i5000_sp0.01_ns0.1",
    "echo_chamber_u500_i5000_sp0.01_ns0.1",
    "fake_users_u500_i5000_sp0.01_ns0.1_fake0.1",
    "line_graph_u500_i5000_sp0.01_ns0.1_max30_decay0.5",
    "partial_alignment_u500_i5000_sp0.01_ns0.1",
    "random_u500_i5000_sp0.01_ns0.1",
    "scale_free_u500_i5000_sp0.01_ns0.1",
    "small_world_u500_i5000_sp0.01_ns0.1",
    "star_u500_i5000_sp0.01_ns0.1",
    "contrarian_u5000_i5000_sp0.01_ns0.1",
    "echo_chamber_u5000_i5000_sp0.01_ns0.1",
    "fake_users_u5000_i5000_sp0.01_ns0.1_fake0.1",
    "line_graph_u5000_i5000_sp0.01_ns0.1_max30_decay0.5",
    "partial_alignment_u5000_i5000_sp0.01_ns0.1",
    "random_u5000_i5000_sp0.01_ns0.1",
    "scale_free_u5000_i5000_sp0.01_ns0.1",
    "small_world_u5000_i5000_sp0.01_ns0.1",
    "star_u5000_i5000_sp0.01_ns0.1",
]

# (datasets, data_seeds, model_seeds)
# - real batches: no data seed dimension → path is mseed_N/dataset/model/
# - synthetic batches: dseed_N/mseed_M/dataset/model/
BATCHES = {
    "DEFAULT_BATCH_REAL": (REAL_DATASETS, [None], [1, 12, 123]),
    "DEFAULT_BATCH_SYNTH": (SYNTH_DATASETS, [1, 2, 3], [1, 12, 123]),
    "TUNED_BATCH_REAL": (REAL_DATASETS, [None], [1, 12, 123]),
    "TUNED_BATCH_SYNTH": (SYNTH_DATASETS, [1, 2, 3], [1, 12, 123]),
}

# marker file for a completed run
DONE_FILE = "final_metrics.csv"


def _run_dir(batch_path, data_seed, model_seed, dataset, model):
    if data_seed is None:
        # real datasets — no data seed dimension
        return os.path.join(batch_path, f"mseed_{model_seed}", dataset, model)
    return os.path.join(batch_path, f"dseed_{data_seed}", f"mseed_{model_seed}", dataset, model)


def check_batch(batch_name, datasets, data_seeds, model_seeds):
    batch_path = os.path.join(RESULTS_DIR, batch_name)
    total = len(datasets) * len(MODELS) * len(data_seeds) * len(model_seeds)
    done, missing, incomplete = [], [], []

    for ds in data_seeds:
        for ms in model_seeds:
            for dataset in datasets:
                for model in MODELS:
                    run_dir = _run_dir(batch_path, ds, ms, dataset, model)
                    marker = os.path.join(run_dir, DONE_FILE)
                    if not os.path.exists(run_dir):
                        missing.append((ds, ms, dataset, model))
                    elif os.path.exists(marker):
                        done.append((ds, ms, dataset, model))
                    else:
                        incomplete.append((ds, ms, dataset, model))

    return total, done, missing, incomplete


def print_batch_report(batch_name, datasets, data_seeds, model_seeds):
    total, done, missing, incomplete = check_batch(batch_name, datasets, data_seeds, model_seeds)
    n_done = len(done)
    n_miss = len(missing)
    n_inc = len(incomplete)

    status = "✅" if n_done == total else ("🔄" if n_done > 0 else "❌")

    print(f"\n{'=' * 60}")
    print(f"{status}  {batch_name}")
    print(f"{'=' * 60}")
    real_only = data_seeds == [None]
    if real_only:
        print(f"  Model seeds    : {model_seeds}")
        print(
            f"  Total expected : {total}  ({len(datasets)} datasets × {len(MODELS)} models × {len(model_seeds)} mseeds)"
        )
    else:
        print(f"  Data seeds     : {data_seeds}")
        print(f"  Model seeds    : {model_seeds}")
        print(
            f"  Total expected : {total}  ({len(datasets)} datasets × {len(MODELS)} models × {len(data_seeds)} dseeds × {len(model_seeds)} mseeds)"
        )
    print(f"  Done           : {n_done}")
    print(f"  Missing        : {n_miss}")
    print(f"  Incomplete     : {n_inc}")

    if missing:
        print("\n  ── Missing (copy for resubmission) ──")
        for ds, ms, dataset, model in missing:
            prefix = f"mseed={ms}" if real_only else f"dseed={ds}  mseed={ms}"
            print(f"    {prefix}  {model:10s}  {dataset}")

    if incomplete:
        print(f"\n  ── Incomplete (ran but no {DONE_FILE}) ──")
        for ds, ms, dataset, model in incomplete:
            prefix = f"mseed={ms}" if real_only else f"dseed={ds}  mseed={ms}"
            print(f"    {prefix}  {model:10s}  {dataset}")

    expected_per_model = len(datasets) * len(data_seeds) * len(model_seeds)
    print("\n  ── Per-model completion ──")
    for model in MODELS:
        m_done = sum(1 for _, _, _, m in done if m == model)
        m_miss = sum(1 for _, _, _, m in missing if m == model)
        m_inc = sum(1 for _, _, _, m in incomplete if m == model)
        bar = "█" * m_done + "░" * (m_miss + m_inc)
        print(
            f"    {model:10s}  {m_done:4d}/{expected_per_model}  [{bar[:40]}]"
            + (f"  ← {m_miss} missing, {m_inc} incomplete" if (m_miss + m_inc) else "")
        )

    return missing, incomplete


def main():
    print(f"\nResults base: {RESULTS_DIR}")
    all_missing, all_incomplete = [], []

    for batch_name, (datasets, data_seeds, model_seeds) in BATCHES.items():
        batch_path = os.path.join(RESULTS_DIR, batch_name)
        if not os.path.exists(batch_path):
            print(f"\n{'=' * 60}")
            print(f"⚪  {batch_name}  — folder not found, skipping")
            continue

        missing, incomplete = print_batch_report(batch_name, datasets, data_seeds, model_seeds)
        all_missing += [(batch_name, ds, ms, d, m) for ds, ms, d, m in missing]
        all_incomplete += [(batch_name, ds, ms, d, m) for ds, ms, d, m in incomplete]

    print(f"\n{'=' * 60}")
    print("GLOBAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total missing across all batches    : {len(all_missing)}")
    print(f"  Total incomplete across all batches : {len(all_incomplete)}")

    if all_missing or all_incomplete:
        out = os.path.join(RESULTS_DIR, "completion_report.txt")
        with open(out, "w") as f:
            f.write("MISSING RUNS\n" + "=" * 60 + "\n")
            for batch, ds, ms, d, m in all_missing:
                f.write(f"{batch}  dseed={ds}  mseed={ms}  {m:10s}  {d}\n")
            f.write("\nINCOMPLETE RUNS\n" + "=" * 60 + "\n")
            for batch, ds, ms, d, m in all_incomplete:
                f.write(f"{batch}  dseed={ds}  mseed={ms}  {m:10s}  {d}\n")
        print(f"\n  Full report saved → {out}")


if __name__ == "__main__":
    main()
