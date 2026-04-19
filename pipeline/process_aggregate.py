"""Collect every final_metrics.csv under results/ into one flat CSV."""

import os
import re

import pandas as pd
import yaml

RESULTS_DIR = "/n/fs/recbench/new_rec/results"
OUT_CSV = "results/all_results.csv"

dseed_pat = re.compile(r"^dseed_(\d+)$")
mseed_pat = re.compile(r"^mseed_(\d+)$")

rows = []

for root, dirs, files in os.walk(RESULTS_DIR):
    if "final_metrics.csv" not in files:
        continue

    # path layout: RESULTS_DIR/batch/[dseed_D/][mseed_M/]dataset/model
    rel = os.path.relpath(root, RESULTS_DIR)
    parts = rel.split(os.sep)
    if len(parts) < 3:
        continue  # malformed / too shallow

    model_name = parts[-1]
    dataset_name = parts[-2]
    batch_folder = parts[0]

    # skip final_comparison output trees
    if "final_comparison" in parts or "final_comparison_baselines" in parts:
        continue

    dseed = None
    mseed = None
    for p in parts[1:-2]:
        m = dseed_pat.match(p)
        if m:
            dseed = int(m.group(1))
            continue
        m = mseed_pat.match(p)
        if m:
            mseed = int(m.group(1))
            continue

    is_tuned = "TUNED" in batch_folder

    final_csv = os.path.join(root, "final_metrics.csv")
    try:
        df = pd.read_csv(final_csv)
    except Exception as e:
        print(f"  ⚠  Could not read {final_csv}: {e}")
        continue

    # folder structure is authoritative over CSV contents
    df = df.rename(columns={"Model": "model", "Dataset": "dataset", "Mode": "mode"})
    df["batch"] = batch_folder
    df["tuned"] = is_tuned
    df["dseed"] = dseed
    df["mseed"] = mseed
    df["dataset"] = dataset_name
    df["model"] = model_name

    if is_tuned:
        hp_yaml = os.path.join(root, "best_hyperparameters.yaml")
        if os.path.exists(hp_yaml):
            try:
                with open(hp_yaml) as f:
                    df["best_params"] = str(yaml.safe_load(f))
            except Exception:
                df["best_params"] = ""
        elif "Best_Params" in df.columns:
            df = df.rename(columns={"Best_Params": "best_params"})
    else:
        df = df.drop(columns=[c for c in ["Best_Params", "mode"] if c in df.columns])

    rows.append(df)

if not rows:
    print("No results found — check RESULTS_DIR")
    raise SystemExit

result = pd.concat(rows, ignore_index=True)
result = result.drop(columns=[c for c in ["Best_Params", "Mode"] if c in result.columns])

front_cols = ["batch", "tuned", "dseed", "mseed", "dataset", "model", "best_params"]
front_cols = [c for c in front_cols if c in result.columns]
metric_cols = [c for c in result.columns if c not in front_cols]
result = result[front_cols + metric_cols]

sort_cols = [c for c in ["batch", "dataset", "model", "dseed", "mseed"] if c in result.columns]
result = result.sort_values(sort_cols).reset_index(drop=True)

result.to_csv(OUT_CSV, index=False)
print(f"Saved {len(result)} rows → {OUT_CSV}")
print(f"  batches : {sorted(result['batch'].unique())}")
print(f"  tuned   : {result['tuned'].value_counts().to_dict()}")
if "dseed" in result.columns:
    print(f"  dseeds  : {sorted(result['dseed'].dropna().unique().tolist())}")
if "mseed" in result.columns:
    print(f"  mseeds  : {sorted(result['mseed'].dropna().unique().tolist())}")
print(f"  models  : {sorted(result['model'].unique())}")
