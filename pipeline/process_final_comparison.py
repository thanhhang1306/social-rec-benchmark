"""Aggregate per-seed final_metrics.csv into cross-seed plots under final_comparison/."""

import os
import re
import textwrap
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

BASE_DIR = "/n/fs/recbench/new_rec"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ERROR_DIR = RESULTS_DIR
MISSING_LOG_FILE = os.path.join(ERROR_DIR, "missing_models.txt")


def _metric_label(col):
    return re.sub(r"^best_valid_", "", col, flags=re.IGNORECASE).upper()


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     Saved: {os.path.relpath(path, RESULTS_DIR)}")


def _available_metrics(df):
    return [
        c
        for c in df.columns
        if any(m in c.lower() for m in ["ndcg", "recall", "mrr", "precision", "hit"])
        and not c.lower().startswith("best_valid")
    ]


def _model_order(df, primary_metric):
    return (
        df.groupby("Model")[primary_metric].mean().sort_values(ascending=False).index.tolist()
    )


def _annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h) or h <= 0:
            continue
        ax.annotate(
            f"{h:.4f}",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )


def _collect_leaf_dirs(batch_path):
    """Map dataset -> {dseed_label: [mseed/dataset paths]} for both real and synth layouts."""
    result = defaultdict(lambda: defaultdict(list))

    for entry in sorted(os.listdir(batch_path)):
        entry_path = os.path.join(batch_path, entry)
        if not os.path.isdir(entry_path) or entry == "final_comparison":
            continue

        # real layout: batch/mseed_M/dataset — bucket as dseed_1 since there's no data-seed axis
        if entry.startswith("mseed_"):
            for ds in sorted(os.listdir(entry_path)):
                ds_path = os.path.join(entry_path, ds)
                if os.path.isdir(ds_path):
                    result[ds]["dseed_1"].append(ds_path)

        # synth layout: batch/dseed_D/mseed_M/dataset
        elif entry.startswith("dseed_"):
            for mseed_entry in sorted(os.listdir(entry_path)):
                if not mseed_entry.startswith("mseed_"):
                    continue
                mseed_path = os.path.join(entry_path, mseed_entry)
                if not os.path.isdir(mseed_path):
                    continue
                for ds in sorted(os.listdir(mseed_path)):
                    ds_path = os.path.join(mseed_path, ds)
                    if os.path.isdir(ds_path):
                        result[ds][entry].append(ds_path)

    return result


def _load_final_metrics(mseed_paths):
    frames = []
    for ds_path in mseed_paths:
        for model_folder in sorted(os.listdir(ds_path)):
            model_path = os.path.join(ds_path, model_folder)
            if not os.path.isdir(model_path) or model_folder in ("final_comparison", "propagation"):
                continue
            csv = os.path.join(model_path, "final_metrics.csv")
            if os.path.exists(csv):
                frames.append(pd.read_csv(csv))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_epoch_data(mseed_paths):
    frames = []
    for ds_path in mseed_paths:
        for model_folder in sorted(os.listdir(ds_path)):
            model_path = os.path.join(ds_path, model_folder)
            if not os.path.isdir(model_path) or model_folder in ("final_comparison", "propagation"):
                continue
            csv = os.path.join(model_path, "epoch_data.csv")
            if os.path.exists(csv):
                frames.append(pd.read_csv(csv))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def chart_model_seed_variance(df, out_dir, dataset_name, dseed_label, n_model_seeds):
    """Fixed data seed, vary model seeds. Error bars = std across model seeds."""
    metrics = _available_metrics(df)
    if not metrics:
        return
    order = _model_order(df, metrics[0])
    ds_display = textwrap.fill(dataset_name.replace("_", " "), width=60)

    for m in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=df,
            x="Model",
            y=m,
            order=order,
            palette="viridis",
            estimator="mean",
            errorbar="sd" if n_model_seeds > 1 else None,
            ax=ax,
        )
        _annotate_bars(ax)
        ax.set_title(
            f"{ds_display}  |  {dseed_label}\n"
            f"{_metric_label(m)}  —  model initialization sensitivity"
            f"  (mean ± sd,  {n_model_seeds} model seed(s))",
            fontsize=11,
        )
        ax.set_xlabel("")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"bar_{m.replace('@', '_')}.png"))


def chart_data_seed_variance(dseed_dfs, out_dir, dataset_name):
    """One bar group per model; each bar is a different data seed (mean over model seeds)."""
    records = []
    for dseed_label, df in dseed_dfs.items():
        if df.empty:
            continue
        metrics = _available_metrics(df)
        avg = df.groupby("Model")[metrics].mean().reset_index()
        avg["Data seed"] = dseed_label
        records.append(avg)
    if not records:
        return
    combined = pd.concat(records, ignore_index=True)
    metrics = _available_metrics(combined)
    if not metrics:
        return
    order = _model_order(combined, metrics[0])
    ds_display = textwrap.fill(dataset_name.replace("_", " "), width=60)

    for m in metrics:
        fig, ax = plt.subplots(figsize=(13, 7))
        sns.barplot(
            data=combined,
            x="Model",
            y=m,
            hue="Data seed",
            order=order,
            palette="Set2",
            ax=ax,
        )
        _annotate_bars(ax)
        ax.set_title(
            f"{ds_display}\n"
            f"{_metric_label(m)}  —  data seed sensitivity"
            f"  (each bar = mean over model seeds)",
            fontsize=11,
        )
        ax.set_xlabel("")
        ax.legend(title="Data seed", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"bar_{m.replace('@', '_')}.png"))


def chart_variance_decomposition(dseed_dfs, out_dir, dataset_name):
    """Side-by-side bars per model per metric: model-seed std vs data-seed std."""
    all_records = []
    for dseed_label, df in dseed_dfs.items():
        if df.empty:
            continue
        metrics = _available_metrics(df)
        for model in df["Model"].unique():
            mdf = df[df["Model"] == model]
            for m in metrics:
                vals = mdf[m].dropna()
                if len(vals) < 1:
                    continue
                all_records.append(
                    {
                        "Model": model,
                        "metric": _metric_label(m),
                        "dseed": dseed_label,
                        "model_seed_std": float(vals.std()) if len(vals) > 1 else 0.0,
                        "model_seed_mean": float(vals.mean()),
                    }
                )
    if not all_records:
        return

    rec_df = pd.DataFrame(all_records)
    # data-seed std = std of per-dseed means across data seeds
    ds_std = (
        rec_df.groupby(["Model", "metric"])["model_seed_mean"]
        .std()
        .reset_index()
        .rename(columns={"model_seed_mean": "data_seed_std"})
    )
    ms_std = (
        rec_df.groupby(["Model", "metric"])["model_seed_std"]
        .mean()
        .reset_index()
        .rename(columns={"model_seed_std": "avg_model_seed_std"})
    )
    var_df = ds_std.merge(ms_std, on=["Model", "metric"])

    ds_display = textwrap.fill(dataset_name.replace("_", " "), width=60)

    for met in var_df["metric"].unique():
        sub = var_df[var_df["metric"] == met].sort_values("avg_model_seed_std", ascending=False)
        x = list(range(len(sub)))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            [xi - w / 2 for xi in x],
            sub["avg_model_seed_std"].fillna(0),
            width=w,
            label="Model seed std  (initialization)",
            color="#2D6A9F",
            alpha=0.85,
        )
        ax.bar(
            [xi + w / 2 for xi in x],
            sub["data_seed_std"].fillna(0),
            width=w,
            label="Data seed std  (graph structure)",
            color="#E07B39",
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Model"].tolist(), rotation=20, ha="right")
        ax.set_ylabel(f"Standard deviation  ({met})")
        ax.set_title(
            f"{ds_display}\n"
            f"{met}  —  variance decomposition: initialization vs graph structure",
            fontsize=11,
        )
        ax.legend()
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"variance_decomp_{met.lower().replace('@', '_')}.png"))


def chart_combined(all_df, out_dir, dataset_name, total_seeds):
    """All seeds pooled: mean ± total std."""
    metrics = _available_metrics(all_df)
    if not metrics:
        return
    order = _model_order(all_df, metrics[0])
    ds_display = textwrap.fill(dataset_name.replace("_", " "), width=60)

    for m in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=all_df,
            x="Model",
            y=m,
            order=order,
            palette="viridis",
            estimator="mean",
            errorbar="sd" if total_seeds > 1 else None,
            ax=ax,
        )
        _annotate_bars(ax)
        ax.set_title(
            f"{ds_display}\n"
            f"{_metric_label(m)}  —  all seeds pooled"
            f"  (mean ± sd,  {total_seeds} total seed(s))",
            fontsize=11,
        )
        ax.set_xlabel("")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"bar_{m.replace('@', '_')}.png"))


def chart_training_curves(dseed_to_paths, out_dir, dataset_name):
    frames = []
    for mseed_paths in dseed_to_paths.values():
        df = _load_epoch_data(mseed_paths)
        if not df.empty:
            frames.append(df)
    if not frames:
        return
    df_epochs = pd.concat(frames, ignore_index=True)
    curve_metrics = [
        c
        for c in df_epochs.columns
        if any(x in c.lower() for x in ["ndcg@10", "recall@10", "train_loss"])
    ]
    ds_display = textwrap.fill(dataset_name.replace("_", " "), width=60)

    for m in curve_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_epochs, x="Epoch", y=m, hue="Model", marker="o", ax=ax)
        ax.set_title(f"{ds_display}\n{_metric_label(m)} over epochs (all seeds avg)", fontsize=11)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"line_{m.replace('@', '_')}.png"))


def _check_completion(batch, dataset, dseed_to_paths):
    """Return (all_complete, total_count, missing_entries)."""
    all_complete = True
    total_count = 0
    missing_entries = []

    for dseed_label, mseed_paths in dseed_to_paths.items():
        for ds_path in mseed_paths:
            mseed_label = os.path.basename(os.path.dirname(ds_path))
            for model_folder in sorted(os.listdir(ds_path)):
                model_path = os.path.join(ds_path, model_folder)
                if not os.path.isdir(model_path) or model_folder in ("final_comparison", "propagation"):
                    continue
                total_count += 1
                has_metrics = os.path.exists(os.path.join(model_path, "final_metrics.csv"))
                has_epoch = os.path.exists(os.path.join(model_path, "epoch_data.csv"))
                if not (has_metrics and has_epoch):
                    all_complete = False
                    loc = f"{batch}/{dseed_label}/{mseed_label}/{dataset}/{model_folder}"
                    reason = (
                        "Missing both"
                        if not has_metrics and not has_epoch
                        else "Missing epoch_data.csv"
                        if has_metrics
                        else "Missing final_metrics.csv"
                    )
                    print(f"  ❌ {loc}  [{reason}]")
                    missing_entries.append(f"{loc} | {reason}")

    return all_complete, total_count, missing_entries


def _seeds_signature(dseed_to_paths):
    parts = []
    for dl, paths in sorted(dseed_to_paths.items()):
        for p in sorted(paths):
            parts.append(f"{dl}:{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}")
    return "\n".join(sorted(parts))


def _should_skip(dataset_comp_dir, dseed_to_paths):
    seeds_file = os.path.join(dataset_comp_dir, "seeds_used.txt")
    if not os.path.exists(seeds_file):
        return False
    with open(seeds_file) as f:
        return f.read().strip() == _seeds_signature(dseed_to_paths)


def _write_seeds_used(dataset_comp_dir, dseed_to_paths):
    os.makedirs(dataset_comp_dir, exist_ok=True)
    with open(os.path.join(dataset_comp_dir, "seeds_used.txt"), "w") as f:
        f.write(_seeds_signature(dseed_to_paths))


total = completed = missing = 0
all_missing_entries = []

for batch in sorted(os.listdir(RESULTS_DIR)):
    batch_path = os.path.join(RESULTS_DIR, batch)
    if not os.path.isdir(batch_path):
        continue

    print(f"\nBatch: {batch}")

    dataset_to_dseeds = _collect_leaf_dirs(batch_path)
    if not dataset_to_dseeds:
        print("  ⚠  No recognized seed dirs found; skipping")
        continue

    batch_comp_root = os.path.join(batch_path, "final_comparison")

    for dataset in sorted(dataset_to_dseeds):
        dseed_to_paths = dataset_to_dseeds[dataset]

        all_complete, model_count, missing_entries = _check_completion(
            batch, dataset, dseed_to_paths
        )

        total += model_count
        completed += model_count - len(missing_entries)
        missing += len(missing_entries)
        all_missing_entries.extend(missing_entries)

        if not all_complete or model_count == 0:
            continue

        dataset_comp_dir = os.path.join(batch_comp_root, dataset)

        if _should_skip(dataset_comp_dir, dseed_to_paths):
            print(f"  ⏭️  Skipping {dataset} — already generated, seeds unchanged")
            continue

        n_dseeds = len(dseed_to_paths)
        total_seeds = sum(len(p) for p in dseed_to_paths.values())
        print(f"\n  ✅ {dataset}  ({n_dseeds} data seed(s), {total_seeds} total seed(s))")

        dseed_dfs = {dl: _load_final_metrics(paths) for dl, paths in dseed_to_paths.items()}
        dseed_dfs = {dl: df for dl, df in dseed_dfs.items() if not df.empty}

        # model-seed variance: one chart set per data seed
        print(f"    → model_seeds/  ({n_dseeds} subfolder(s))")
        for dseed_label, df in dseed_dfs.items():
            n_model_seeds = len(dseed_to_paths[dseed_label])
            chart_model_seed_variance(
                df,
                os.path.join(dataset_comp_dir, "model_seeds", dseed_label),
                dataset,
                dseed_label,
                n_model_seeds,
            )

        # data-seed variance and variance decomposition only when >1 data seed present
        if n_dseeds > 1:
            print("    → data_seeds/")
            chart_data_seed_variance(
                dseed_dfs, os.path.join(dataset_comp_dir, "data_seeds"), dataset
            )
            print("    → variance_decomposition/")
            chart_variance_decomposition(
                dseed_dfs, os.path.join(dataset_comp_dir, "variance_decomposition"), dataset
            )

        print("    → combined/")
        all_df = pd.concat(dseed_dfs.values(), ignore_index=True)
        combined_dir = os.path.join(dataset_comp_dir, "combined")
        chart_combined(all_df, combined_dir, dataset, total_seeds)
        chart_training_curves(dseed_to_paths, combined_dir, dataset)

        _write_seeds_used(dataset_comp_dir, dseed_to_paths)

print(f"\nCompleted : {completed}/{total}")
print(f"Missing   : {missing}/{total}")

if all_missing_entries:
    os.makedirs(ERROR_DIR, exist_ok=True)
    with open(MISSING_LOG_FILE, "w") as f:
        f.write("\n".join(all_missing_entries))
    print(f"Missing log  : {MISSING_LOG_FILE}")
else:
    if os.path.exists(MISSING_LOG_FILE):
        os.remove(MISSING_LOG_FILE)
    if total > 0:
        print("All models complete — no missing log.")
