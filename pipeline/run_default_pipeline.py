"""Run a single model/dataset with default (untuned) hyperparameters."""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "bool_"):
    np.bool_ = np.bool8

gnn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RecBole-GNN")
if gnn_path not in sys.path:
    sys.path.append(gnn_path)

import scipy.sparse

if not hasattr(scipy.sparse.dok_matrix, "_update"):
    scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update

GNN_MODELS = {
    "SimGCL",
    "DiffNet",
    "MHCN",
    "LightGCN",
    "HAGN",
    "HAGNUniform",
    "HAGNFixed",
    "HAGNGlobal",
    "CSGCN",
    "CSGCNNoWarmup",
    "ASLGCN",
    "ASLGCNSymmetric",
}


def get_nested_path(model_name, dataset_name, manual_batch="", model_seed=2020, data_seed=1):
    base_results = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    mode = "DEFAULT"
    os.makedirs(base_results, exist_ok=True)
    if manual_batch:
        current_batch = manual_batch
    else:
        existing = [d for d in os.listdir(base_results) if d.startswith(f"{mode}_BATCH_")]
        batch_nums = [int(d.split("_")[-1]) for d in existing if d.split("_")[-1].isdigit()]
        current_batch = f"{mode}_BATCH_{max(batch_nums)}" if batch_nums else f"{mode}_BATCH_1"
    real_datasets_list = ["yelp", "douban-book", "lastfm"]
    if dataset_name in real_datasets_list:
        path = os.path.join(
            base_results, current_batch, f"mseed_{model_seed}", dataset_name, model_name
        )
    else:
        path = os.path.join(
            base_results,
            current_batch,
            f"dseed_{data_seed}",
            f"mseed_{model_seed}",
            dataset_name,
            model_name,
        )
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup_checkpoints(run_dir):
    """Delete .pth checkpoint files after a successful run to save disk space."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return
    removed = 0
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth"):
            os.remove(os.path.join(ckpt_dir, f))
            removed += 1
    if removed:
        print(f"  [disk] Removed {removed} checkpoint file(s) from {ckpt_dir}")


def _patch_trainer_for_epoch_tracking(trainer):
    """Capture per-epoch validation metrics and epoch wall time."""
    trainer._epoch_valid_results = {}
    trainer._epoch_times = {}
    _orig_train = trainer._train_epoch
    _orig_valid = trainer._valid_epoch
    _count = [0]
    _t_epoch_start = [None]

    def _patched_train(*args, **kwargs):
        _t_epoch_start[0] = time.time()
        return _orig_train(*args, **kwargs)

    def _patched_valid(*args, **kwargs):
        score, result = _orig_valid(*args, **kwargs)
        ep = _count[0]
        trainer._epoch_valid_results[ep] = {
            "valid_score": score,
            **{k.lower(): v for k, v in result.items()},
        }
        if _t_epoch_start[0] is not None:
            trainer._epoch_times[ep] = time.time() - _t_epoch_start[0]
        _count[0] += 1
        return score, result

    trainer._train_epoch = _patched_train
    trainer._valid_epoch = _patched_valid


def run_default_pipeline(model_name, dataset_name, batch_name, model_seed=2020, data_seed=1):
    run_dir = os.path.normpath(
        get_nested_path(model_name, dataset_name, batch_name, model_seed, data_seed)
    )
    print(f"THESIS_ID: {dataset_name}_{model_name}_{run_dir}", flush=True)

    if os.path.exists(os.path.join(run_dir, "final_metrics.csv")):
        print(f"⏭️  Skipping {model_name} on {dataset_name} — final_metrics.csv already exists.")
        return

    real_datasets_list = ["yelp", "douban-book", "lastfm"]
    if dataset_name in real_datasets_list:
        data_path = "./datasets"
    else:
        data_path = f"./datasets/dseed_{data_seed}"

    config_dict = {
        "data_path": data_path,
        "checkpoint_dir": os.path.join(run_dir, "checkpoints"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": None,
        "net_file_extension": ".net",
        "social_id_field": "source_id",
        "fields_in_same_space": [["user_id", "source_id", "target_id"]],
        "load_col": {"inter": ["user_id", "item_id"], "net": ["source_id", "target_id"]},
        "user_inter_num_interval": "[10,inf)",
        "item_inter_num_interval": "[10,inf)",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": {"valid": "full", "test": "full"},
        },
        "embedding_size": 64,
        "n_layers": 3,
        "train_batch_size": 4096,
        "eval_batch_size": 40960000,
        "epochs": 500,
        "stopping_step": 10,
        "eval_step": 1,
        "valid_metric": "NDCG@10",
        "metrics": ["Recall", "NDCG", "MRR", "Precision", "Hit"],
        "topk": [10, 20],
        "seed": model_seed,
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1, "dynamic": False},
    }

    if model_name in ("HAGN", "HAGNUniform", "HAGNFixed", "HAGNGlobal"):
        config_dict["require_pow"] = False

    elif model_name in ("CSGCN", "CSGCNNoWarmup"):
        config_dict["warmup_epochs"] = 0 if model_name == "CSGCNNoWarmup" else 20
        config_dict["anneal_epochs"] = 30
        config_dict["require_pow"] = False

    elif model_name in ("ASLGCN", "ASLGCNSymmetric"):
        config_dict["social_layers"] = 1
        config_dict["require_pow"] = False

    real_datasets = ["yelp", "douban-book", "lastfm"]

    if dataset_name not in real_datasets:
        config_dict["user_inter_num_interval"] = "[3,inf)"
        config_dict["item_inter_num_interval"] = "[1,inf)"
        print(f"[{dataset_name}] Detected synthetic data: Disabled 10-core filtering.")
    else:
        print(f"[{dataset_name}] Detected real data: Applying [10,inf) core filtering.")

    inter_file_path = os.path.join(data_path, dataset_name, f"{dataset_name}.inter")
    if os.path.exists(inter_file_path) and dataset_name in real_datasets:
        with open(inter_file_path, "r") as f:
            if "rating" in f.readline().lower():
                config_dict.update(
                    {
                        "RATING_FIELD": "rating",
                        "val_interval": {"rating": "[3,inf)"},
                        "unused_col": {"inter": ["rating"]},
                    }
                )
                config_dict["load_col"]["inter"].append("rating")

    config_yaml = os.path.join(run_dir, "config.yaml")
    with open(config_yaml, "w") as f:
        yaml.dump(config_dict, f)

    try:
        sys.argv = [sys.argv[0], f"--model={model_name}", f"--dataset={dataset_name}"]

        inference_time_s = None
        train_time_s = None
        n_params = None

        if model_name == "HAGN":
            from recbole.trainer import Trainer
            from recbole_gnn.config import Config
            from recbole_gnn.model.social_recommender.hagn import HAGN
            from recbole_gnn.utils import create_dataset, data_preparation

            cfg = Config(
                model=HAGN,
                dataset=dataset_name,
                config_dict=config_dict,
                config_file_list=[config_yaml],
            )
            ds = create_dataset(cfg)
            train_data, valid_data, test_data = data_preparation(cfg, ds)

            model = HAGN(cfg, ds).to(cfg["device"])
            n_params = sum(p.numel() for p in model.parameters())
            trainer = Trainer(cfg, model)
            _patch_trainer_for_epoch_tracking(trainer)

            t_train = time.time()
            _, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=True
            )
            train_time_s = time.time() - t_train
            t0 = time.time()
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
            inference_time_s = time.time() - t0
            result = {"best_valid_result": best_valid_result, "test_result": test_result}

        elif model_name in ("CSGCN", "CSGCNNoWarmup"):
            from recbole_gnn.config import Config
            from recbole_gnn.model.social_recommender.csgcn import CSGCN, CSGCNTrainer
            from recbole_gnn.utils import create_dataset, data_preparation

            cfg = Config(
                model=CSGCN,
                dataset=dataset_name,
                config_dict=config_dict,
                config_file_list=[config_yaml],
            )
            ds = create_dataset(cfg)
            train_data, valid_data, test_data = data_preparation(cfg, ds)

            model = CSGCN(cfg, ds).to(cfg["device"])
            n_params = sum(p.numel() for p in model.parameters())
            trainer = CSGCNTrainer(cfg, model)
            _patch_trainer_for_epoch_tracking(trainer)

            t_train = time.time()
            _, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=True
            )
            train_time_s = time.time() - t_train
            t0 = time.time()
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
            inference_time_s = time.time() - t0
            result = {"best_valid_result": best_valid_result, "test_result": test_result}

        elif model_name == "ASLGCN":
            from recbole.trainer import Trainer
            from recbole_gnn.config import Config
            from recbole_gnn.model.social_recommender.aslgcn import ASLGCN
            from recbole_gnn.utils import create_dataset, data_preparation

            cfg = Config(
                model=ASLGCN,
                dataset=dataset_name,
                config_dict=config_dict,
                config_file_list=[config_yaml],
            )
            ds = create_dataset(cfg)
            train_data, valid_data, test_data = data_preparation(cfg, ds)

            model = ASLGCN(cfg, ds).to(cfg["device"])
            n_params = sum(p.numel() for p in model.parameters())
            trainer = Trainer(cfg, model)
            _patch_trainer_for_epoch_tracking(trainer)

            t_train = time.time()
            _, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=True
            )
            train_time_s = time.time() - t_train
            t0 = time.time()
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
            inference_time_s = time.time() - t0
            result = {"best_valid_result": best_valid_result, "test_result": test_result}

        elif model_name in GNN_MODELS:
            from recbole_gnn.config import Config
            from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

            cfg = Config(
                model=model_name,
                dataset=dataset_name,
                config_dict=config_dict,
                config_file_list=[config_yaml],
            )
            ds = create_dataset(cfg)
            train_data, valid_data, test_data = data_preparation(cfg, ds)

            model_class = get_model(cfg["model"])
            model = model_class(cfg, ds).to(cfg["device"])
            n_params = sum(p.numel() for p in model.parameters())
            trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)
            _patch_trainer_for_epoch_tracking(trainer)

            t_train = time.time()
            _, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=True
            )
            train_time_s = time.time() - t_train
            t0 = time.time()
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
            inference_time_s = time.time() - t0
            result = {"best_valid_result": best_valid_result, "test_result": test_result}

        else:
            from recbole.config import Config as RecBoleConfig
            from recbole.data import create_dataset as rb_create_dataset
            from recbole.data import data_preparation as rb_data_preparation
            from recbole.utils import get_model as get_recbole_model
            from recbole.utils import get_trainer as get_recbole_trainer

            cfg = RecBoleConfig(
                model=model_name,
                dataset=dataset_name,
                config_dict=config_dict,
                config_file_list=[config_yaml],
            )
            ds = rb_create_dataset(cfg)
            train_data, valid_data, test_data = rb_data_preparation(cfg, ds)

            model_class = get_recbole_model(model_name)
            model = model_class(cfg, ds).to(cfg["device"])
            n_params = sum(p.numel() for p in model.parameters())
            trainer = get_recbole_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)
            _patch_trainer_for_epoch_tracking(trainer)

            t_train = time.time()
            _, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=True
            )
            train_time_s = time.time() - t_train
            t0 = time.time()
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
            inference_time_s = time.time() - t0
            result = {"best_valid_result": best_valid_result, "test_result": test_result}

        final_metrics = result.get("test_result", result) if isinstance(result, dict) else result
        best_valid_ndcg = (
            best_valid_result.get("ndcg@10") if isinstance(best_valid_result, dict) else None
        )

        dataset_type = "real" if dataset_name in real_datasets else "synth"
        total_epochs = len(trainer.train_loss_dict) if hasattr(trainer, "train_loss_dict") else 0

        best_epoch = None
        if hasattr(trainer, "_epoch_valid_results") and trainer._epoch_valid_results:
            best_idx = max(
                trainer._epoch_valid_results,
                key=lambda k: trainer._epoch_valid_results[k]["valid_score"],
            )
            best_epoch = best_idx + 1  # epoch keys are 0-indexed

        row_data = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Mode": "DEFAULT",
            "dataset_type": dataset_type,
            "seed": model_seed,
            "data_seed": data_seed,
            "total_epochs_trained": total_epochs,
            "best_epoch": best_epoch,
            "n_parameters": n_params,
            "train_time_s": train_time_s,
            "inference_time_s": inference_time_s,
            "best_valid_ndcg@10": best_valid_ndcg,
        }

        for k, v in final_metrics.items():
            if isinstance(v, (int, float, np.float64)):
                row_data[k] = v

        if hasattr(trainer, "train_loss_dict") and trainer.train_loss_dict:
            epoch_rows = []
            for ep, loss in sorted(trainer.train_loss_dict.items()):
                row = {"Model": model_name, "Epoch": ep + 1, "train_loss": loss}
                if hasattr(trainer, "_epoch_times") and ep in trainer._epoch_times:
                    row["epoch_time_s"] = trainer._epoch_times[ep]
                if hasattr(trainer, "_epoch_valid_results") and ep in trainer._epoch_valid_results:
                    row.update(trainer._epoch_valid_results[ep])
                epoch_rows.append(row)
            pd.DataFrame(epoch_rows).to_csv(os.path.join(run_dir, "epoch_data.csv"), index=False)

        res_df = pd.DataFrame([row_data])
        res_df.to_csv(os.path.join(run_dir, "final_metrics.csv"), index=False)

        latex_vals = [
            f"{res_df.iloc[0][c]:.4f}"
            if isinstance(res_df.iloc[0].get(c), (float, np.float64))
            else str(res_df.iloc[0].get(c, "N/A"))
            for c in res_df.columns
        ]
        with open(os.path.join(run_dir, "latex_row.tex"), "w") as f:
            f.write("% " + " & ".join(res_df.columns) + " \\\\\n")
            f.write(" & ".join(latex_vals) + " \\\\\n")

        print(f"✅ Success: {run_dir}")
        _cleanup_checkpoints(run_dir)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ FAILURE: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch", type=str, default="")
    parser.add_argument("--seed", type=int, default=2020, help="Model/optimizer random seed")
    parser.add_argument(
        "--data_seed", type=int, default=1, help="Data seed (1=original ./datasets/)"
    )
    args = parser.parse_args()
    run_default_pipeline(args.model, args.dataset, args.batch, args.seed, args.data_seed)
