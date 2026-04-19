"""Per-user metrics vs chain position on line_graph datasets."""

import argparse
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

warnings.filterwarnings("ignore")

# fix numpy deprecations for RecBole compatibility
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "bool_"):
    np.bool_ = np.bool8

GNN_MODELS = {
    "SimGCL",
    "DiffNet",
    "MHCN",
    "LightGCN",
    "HAGN",
    "HAGNUniform",
    "HAGNFixed",
    "CSGCN",
    "CSGCNNoWarmup",
    "ASLGCN",
    "ASLGCNSymmetric",
}

HYBRID_MODELS = {"HAGN", "CSGCN", "ASLGCN"}

ALL_MODELS = ["BPR", "LightGCN", "SimGCL", "DiffNet", "MHCN", "HAGN", "CSGCN", "ASLGCN"]

TOPK = [10, 20]

BASE_DIR = "/n/fs/recbench/new_rec"
AGGREGATE_CSV = os.path.join(BASE_DIR, "results", "propagation_all.csv")


def get_run_dir(dataset, model, batch, dseed, mseed):
    return os.path.join(
        BASE_DIR, "results", batch, f"dseed_{dseed}", f"mseed_{mseed}", dataset, model
    )


def load_best_config(run_dir):
    """Load config.yaml and overlay best_hyperparameters.yaml."""
    config_path = os.path.join(run_dir, "config.yaml")
    hp_path = os.path.join(run_dir, "best_hyperparameters.yaml")

    if not os.path.exists(config_path) or not os.path.exists(hp_path):
        return None

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    with open(hp_path) as f:
        best_params = yaml.safe_load(f)

    config_dict.update(best_params)
    config_dict["show_progress"] = True
    return config_dict


def _per_user_ndcg(pos_index, pos_len, k):
    pi = pos_index[:, :k]
    pl = pos_len.squeeze()
    ranks = np.arange(1, k + 1)
    dcg = np.cumsum(pi / np.log2(ranks + 1), axis=1)[:, -1]
    idcg_len = np.minimum(pl, k).astype(int)
    idcg = np.zeros(len(pl))
    for i, length in enumerate(idcg_len):
        if length > 0:
            idcg[i] = np.sum(1.0 / np.log2(np.arange(1, length + 1) + 1))
    return np.where(idcg > 0, dcg / idcg, 0.0)


def _per_user_recall(pos_index, pos_len, k):
    hits = pos_index[:, :k].sum(axis=1)
    pl = pos_len.squeeze().astype(float)
    return np.where(pl > 0, hits / pl, 0.0)


def _per_user_precision(pos_index, pos_len, k):
    return pos_index[:, :k].sum(axis=1) / k


def _per_user_hit(pos_index, pos_len, k):
    return (pos_index[:, :k].sum(axis=1) > 0).astype(float)


def _per_user_mrr(pos_index, pos_len, k):
    pi = pos_index[:, :k]
    # first hit position (1-indexed), 0 if no hit
    first_hit = np.argmax(pi, axis=1) + 1
    has_hit = pi.sum(axis=1) > 0
    return np.where(has_hit, 1.0 / first_hit, 0.0)


def compute_per_user_metrics(pos_index, pos_len):
    """Return {metric_name: per_user_array} across all TOPK values."""
    metrics = {}
    for k in TOPK:
        metrics[f"ndcg@{k}"] = _per_user_ndcg(pos_index, pos_len, k)
        metrics[f"recall@{k}"] = _per_user_recall(pos_index, pos_len, k)
        metrics[f"precision@{k}"] = _per_user_precision(pos_index, pos_len, k)
        metrics[f"hit@{k}"] = _per_user_hit(pos_index, pos_len, k)
        metrics[f"mrr@{k}"] = _per_user_mrr(pos_index, pos_len, k)
    return metrics


def train_and_eval_per_user(model_name, dataset_name, config_dict):
    """Train model and return per-user metrics with original user IDs."""
    gnn_path = os.path.join(BASE_DIR, "RecBole-GNN")
    if gnn_path not in sys.path:
        sys.path.insert(0, gnn_path)

    import scipy.sparse

    if not hasattr(scipy.sparse.dok_matrix, "_update"):
        scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update

    try:
        import recbole_gnn.model  # noqa: F401
    except ImportError:
        pass

    if model_name in ("CSGCN", "CSGCNNoWarmup"):
        from recbole_gnn.config import Config
        from recbole_gnn.model.social_recommender.csgcn import CSGCN, CSGCNTrainer
        from recbole_gnn.utils import create_dataset, data_preparation

        cfg = Config(model=CSGCN, dataset=dataset_name, config_dict=config_dict)
        ds = create_dataset(cfg)
        train_data, valid_data, test_data = data_preparation(cfg, ds)
        model = CSGCN(cfg, ds).to(cfg["device"])
        trainer = CSGCNTrainer(cfg, model)

    elif model_name == "ASLGCN":
        from recbole_gnn.config import Config
        from recbole_gnn.model.social_recommender.aslgcn import ASLGCN
        from recbole_gnn.utils import create_dataset, data_preparation, get_trainer

        cfg = Config(model=ASLGCN, dataset=dataset_name, config_dict=config_dict)
        ds = create_dataset(cfg)
        train_data, valid_data, test_data = data_preparation(cfg, ds)
        model = ASLGCN(cfg, ds).to(cfg["device"])
        trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    elif model_name in GNN_MODELS:
        from recbole_gnn.config import Config
        from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

        cfg = Config(model=model_name, dataset=dataset_name, config_dict=config_dict)
        ds = create_dataset(cfg)
        train_data, valid_data, test_data = data_preparation(cfg, ds)
        model_class = get_model(cfg["model"])
        model = model_class(cfg, ds).to(cfg["device"])
        trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    else:  # BPR etc.
        from recbole.config import Config as RecBoleConfig
        from recbole.data import create_dataset as rb_create_dataset
        from recbole.data import data_preparation as rb_data_preparation
        from recbole.utils import get_model as get_recbole_model
        from recbole.utils import get_trainer as get_recbole_trainer

        cfg = RecBoleConfig(model=model_name, dataset=dataset_name, config_dict=config_dict)
        ds = rb_create_dataset(cfg)
        train_data, valid_data, test_data = rb_data_preparation(cfg, ds)
        model_class = get_recbole_model(model_name)
        model = model_class(cfg, ds).to(cfg["device"])
        trainer = get_recbole_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    print(f"  Training {model_name}...")
    t0 = time.time()
    trainer.fit(train_data, valid_data, saved=True, show_progress=True)
    print(f"  Trained in {time.time() - t0:.0f}s")

    print("  Evaluating per-user...")
    model.eval()

    eval_func = trainer._full_sort_batch_eval
    if trainer.item_tensor is None:
        trainer.item_tensor = test_data._dataset.get_item_feature().to(cfg["device"])
    trainer.tot_item_num = test_data._dataset.item_num

    all_topk = []
    all_user_ids = []

    max_k = max(TOPK)

    with torch.no_grad():
        for batch_idx, batched_data in enumerate(test_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)

            _, topk_idx = torch.topk(scores, max_k, dim=-1)
            pos_matrix = torch.zeros_like(scores, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            all_topk.append(result.cpu().numpy())

            uid_field = cfg["USER_ID_FIELD"]
            batch_uids = interaction[uid_field].cpu().numpy()
            all_user_ids.append(batch_uids)

    topk_matrix = np.concatenate(all_topk, axis=0)
    user_ids_internal = np.concatenate(all_user_ids, axis=0)

    pos_index = topk_matrix[:, :-1]  # (n_users, max_k)
    pos_len = topk_matrix[:, -1:]  # (n_users, 1)

    metrics = compute_per_user_metrics(pos_index, pos_len)

    # internal RecBole IDs map back to original token IDs which index chain position
    uid_field = cfg["USER_ID_FIELD"]
    id2token = ds.field2id_token[uid_field]
    original_ids = np.array([int(id2token[uid]) for uid in user_ids_internal])

    import glob

    ckpt_dir = config_dict.get("checkpoint_dir", "")
    if ckpt_dir and os.path.exists(ckpt_dir):
        for f in glob.glob(os.path.join(ckpt_dir, "*.pth")):
            os.remove(f)

    # clear logging handlers to avoid accumulation across repeated runs
    import logging

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    return original_ids, metrics


def make_plots(df, out_dir, dataset_name, n_bins_list=(10, 20, 50)):
    """Generate propagation plots for all metrics and bin sizes."""
    sns.set_theme(style="whitegrid", context="talk")

    n_users = df["chain_position"].max() + 1
    df = df.copy()
    df["position_ratio"] = df["chain_position"] / (n_users - 1)

    metric_cols = [c for c in df.columns if "@" in c]

    for n_bins in n_bins_list:
        df["bin"] = pd.cut(df["position_ratio"], bins=n_bins, labels=False)
        df["bin_center"] = df["bin"] / n_bins + 0.5 / n_bins

        bin_dir = os.path.join(out_dir, f"bins_{n_bins}")
        os.makedirs(bin_dir, exist_ok=True)

        for metric in metric_cols:
            for suffix, exclude in [("", set()), ("_baselines", HYBRID_MODELS)]:
                plot_df = df[~df["model"].isin(exclude)] if exclude else df

                fig, ax = plt.subplots(figsize=(14, 7))
                binned = (
                    plot_df.groupby(["model", "bin_center"])[metric]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )

                for model_name in sorted(plot_df["model"].unique()):
                    m = binned[binned["model"] == model_name]
                    ax.plot(
                        m["bin_center"],
                        m["mean"],
                        marker="o",
                        markersize=3,
                        label=model_name,
                        linewidth=2,
                    )
                    ax.fill_between(
                        m["bin_center"], m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.1
                    )

                ax.set_xlabel("Chain position ratio\n(0 = sparse end, 1 = rich end)")
                ax.set_ylabel(metric.upper())
                ax.set_title(
                    f"{dataset_name}\n{metric.upper()} vs chain position"
                    f" ({n_bins} bins, {n_users} users)"
                )
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                ax.set_xlim(-0.02, 1.02)
                fig.tight_layout()

                safe_metric = metric.replace("@", "_")
                fname = f"propagation_{safe_metric}{suffix}.png"
                fig.savefig(os.path.join(bin_dir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)

        print(f"  Saved plots: {bin_dir}/ ({len(metric_cols)} metrics × 2 variants)")


def append_to_aggregate(df, dataset, dseed, mseed):
    """Append results to the central propagation_all.csv."""
    df = df.copy()
    df["dataset"] = dataset
    df["dseed"] = dseed
    df["mseed"] = mseed

    if os.path.exists(AGGREGATE_CSV):
        existing = pd.read_csv(AGGREGATE_CSV)
        # drop existing rows for this dataset+dseed+mseed to avoid duplicates
        mask = ~(
            (existing["dataset"] == dataset)
            & (existing["dseed"] == dseed)
            & (existing["mseed"] == mseed)
        )
        existing = existing[mask]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_csv(AGGREGATE_CSV, index=False)
    print(f"  Aggregate: {len(combined)} total rows → {AGGREGATE_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Line graph propagation analysis: per-user metrics vs chain position"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dseed", type=int, default=1)
    parser.add_argument("--mseed", type=int, default=1)
    parser.add_argument("--batch", type=str, default="TUNED_BATCH_SYNTH")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Models to evaluate (default: all 8)"
    )
    parser.add_argument(
        "--bins",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="Bin counts for plots (default: 10 20 50)",
    )
    args = parser.parse_args()

    models = args.models or ALL_MODELS

    out_dir = os.path.join(
        BASE_DIR,
        "results",
        args.batch,
        f"dseed_{args.dseed}",
        f"mseed_{args.mseed}",
        args.dataset,
        "propagation",
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "per_user_metrics.csv")

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_models = set(existing["model"].unique())
        remaining = [m for m in models if m not in done_models]
        if not remaining:
            print("All models already evaluated. Regenerating plots only.")
            make_plots(existing, out_dir, args.dataset, args.bins)
            append_to_aggregate(existing, args.dataset, args.dseed, args.mseed)
            return
        print(f"Already done: {sorted(done_models)}. Remaining: {remaining}")
        models = remaining
    else:
        existing = pd.DataFrame()

    all_rows = []

    for model_name in models:
        run_dir = get_run_dir(args.dataset, model_name, args.batch, args.dseed, args.mseed)

        config_dict = load_best_config(run_dir)
        if config_dict is None:
            print(f"  ⚠  Skipping {model_name} — no config/best_params in {run_dir}")
            continue

        config_dict["checkpoint_dir"] = os.path.join(out_dir, f"ckpt_{model_name}")

        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}  Dataset: {args.dataset}")
        print(f"  dseed={args.dseed}  mseed={args.mseed}")
        print(f"{'=' * 60}")

        try:
            original_ids, metrics = train_and_eval_per_user(model_name, args.dataset, config_dict)

            for i, uid in enumerate(original_ids):
                row = {
                    "model": model_name,
                    "chain_position": uid,
                }
                for metric_name, values in metrics.items():
                    row[metric_name] = values[i]
                all_rows.append(row)

            mean_ndcg = np.mean(metrics["ndcg@10"])
            print(f"  ✅ {model_name}: mean NDCG@10 = {mean_ndcg:.4f}")

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ❌ {model_name} failed: {e}")

    if all_rows:
        new_df = pd.DataFrame(all_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False)
        print(f"\nSaved {len(combined)} rows → {csv_path}")

        make_plots(combined, out_dir, args.dataset, args.bins)
        append_to_aggregate(combined, args.dataset, args.dseed, args.mseed)
    elif not existing.empty:
        make_plots(existing, out_dir, args.dataset, args.bins)
        append_to_aggregate(existing, args.dataset, args.dseed, args.mseed)
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
