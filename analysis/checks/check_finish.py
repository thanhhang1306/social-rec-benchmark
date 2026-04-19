"""Count runs whose final_metrics.csv and epoch_data.csv both exist."""

import os

RESULTS_DIR = "/n/fs/recbench/new_rec/results"

total = 0
completed = 0
missing = 0

for root, dirs, files in os.walk(RESULTS_DIR):
    # skip non-leaf dirs: leaf dirs contain final_metrics.csv or epoch_data.csv
    if "final_metrics.csv" not in files and "epoch_data.csv" not in files:
        continue

    rel = os.path.relpath(root, RESULTS_DIR)

    # skip housekeeping dirs
    if "final_comparison" in rel:
        continue

    total += 1
    has_metrics = "final_metrics.csv" in files
    has_epoch = "epoch_data.csv" in files

    if has_metrics and has_epoch:
        completed += 1
    else:
        missing += 1
        missing_files = []
        if not has_metrics:
            missing_files.append("final_metrics.csv")
        if not has_epoch:
            missing_files.append("epoch_data.csv")
        print(f"❌ {rel}  (missing: {', '.join(missing_files)})")

print(f"\n{'=' * 60}")
print(f"✅ Completed: {completed}/{total}")
print(f"❌ Missing:   {missing}/{total}")
