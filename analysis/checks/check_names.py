"""Verify expected synthetic dataset folders and files exist."""

import os

DATASETS_DIR = "/n/fs/recbench/new_rec/datasets/dseed_3"

datasets = [
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

missing_folders = []
missing_inter = []
missing_net = []

for ds in datasets:
    ds_path = os.path.join(DATASETS_DIR, ds)
    inter_file = os.path.join(ds_path, f"{ds}.inter")
    net_file = os.path.join(ds_path, f"{ds}.net")

    if not os.path.exists(ds_path):
        missing_folders.append(ds)
    else:
        if not os.path.exists(inter_file):
            missing_inter.append(ds)
        if not os.path.exists(net_file):
            missing_net.append(ds)

print(f"Total datasets expected: {len(datasets)}")
print("=" * 60)

if not missing_folders and not missing_inter and not missing_net:
    print("\n✅ All 176 datasets are generated and ready!")
else:
    if missing_folders:
        print(f"\n❌ MISSING FOLDERS ({len(missing_folders)}):")
        for ds in missing_folders:
            print(f"   - {ds}")

    if missing_inter:
        print(f"\n⚠  MISSING .inter FILE ({len(missing_inter)}):")
        for ds in missing_inter:
            print(f"   - {ds}")

    if missing_net:
        print(f"\n⚠  MISSING .net FILE ({len(missing_net)}):")
        for ds in missing_net:
            print(f"   - {ds}")

    print(
        "\n❌ Some datasets are incomplete or missing — run the generation bash script before submitting."
    )
