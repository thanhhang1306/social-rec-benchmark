"""Run exhaustive hyperparameter search via RecBole HyperTuning, then evaluate the best config."""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "bool_"):
    np.bool_ = np.bool8

from recbole.trainer.hyper_tuning import HyperTuning

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
    mode = "TUNED"
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


def run_pipeline(model_name, dataset_name, batch_name, model_seed=2020, data_seed=1):
    run_dir = os.path.normpath(
        get_nested_path(model_name, dataset_name, batch_name, model_seed, data_seed)
    )
    print(f"THESIS_ID: {dataset_name}_{model_name}_{run_dir}", flush=True)

    if os.path.exists(os.path.join(run_dir, "final_metrics.csv")):
        print(f"⏭️  Skipping {model_name} on {dataset_name} — final_metrics.csv already exists.")
        return

    def thesis_objective_function(config_dict=None, config_file_list=None, saved=False):
        import sys

        if not hasattr(sys.stdout, "encoding"):
            sys.stdout.encoding = "utf-8"
        if not hasattr(sys.stderr, "encoding"):
            sys.stderr.encoding = "utf-8"

        gnn_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RecBole-GNN"
        )
        if gnn_path not in sys.path:
            sys.path.append(gnn_path)

        import scipy.sparse

        if not hasattr(scipy.sparse.dok_matrix, "_update"):
            scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update

        try:
            import recbole_gnn.model  # noqa: F401
        except ImportError:
            pass

        if config_dict is None:
            config_dict = {}
        config_dict["model"] = model_name
        config_dict["dataset"] = dataset_name
        config_dict["show_progress"] = False

        if model_name in GNN_MODELS:
            from recbole_gnn.quick_start import objective_function as gnn_objective_function

            return gnn_objective_function(config_dict, config_file_list, saved)
        else:
            from recbole.quick_start import objective_function as base_objective_function

            return base_objective_function(config_dict, config_file_list, saved)

    hp_config = {"learning_rate": [0.005, 0.001, 0.0005, 0.0001], "reg_weight": [1e-3, 1e-4, 1e-5]}
    if model_name == "BPR":
        # BPR doesn't read reg_weight — regularization goes through the optimizer's
        # weight_decay (L2); RecBole's default weight_decay=0.0 leaves BPR unregularized,
        # so search over weight_decay instead
        hp_config = {
            "learning_rate": [0.005, 0.001, 0.0005, 0.0001],
            "weight_decay": [1e-3, 1e-4, 1e-5, 0.0],
        }
        # 4*4 = 16
    elif model_name == "LightGCN":
        hp_config.update({"n_layers": [2, 3, 4], "reg_weight": [1e-3, 1e-4, 1e-5]})
    elif model_name == "SimGCL":
        # reduced from 400 based on preliminary results:
        # lr always 0.001 or 0.0005; temp never 0.05; lambda varies widely
        hp_config["learning_rate"] = [0.001, 0.0005]
        hp_config["reg_weight"] = [1e-4]
        hp_config.update(
            {
                "n_layers": [2, 3],
                "temperature": [0.2],
                "lambda": [0.1, 0.5, 1.0, 2.0],
                "eps": [0.1, 0.2],
            }
        )
        # 2*1*2*2*4*2 = 64
    elif model_name == "DiffNet":
        hp_config.update({"n_layers": [1, 2, 3], "reg_weight": [1e-3, 1e-4, 1e-5]})
    elif model_name == "MHCN":
        # RecBole sums (not means) ss_loss over users, so effective ssl_reg scales ~1/n_users
        # relative to the paper — the paper's optimal ssl_reg=0.01 with a mean-normalized
        # ss_loss maps to ~1e-5..1e-4 here
        hp_config["learning_rate"] = [0.001, 0.0001]
        hp_config.update(
            {
                "n_layers": [1, 2, 3],
                "ssl_reg": [1e-6, 1e-5, 5e-5, 1e-4],
                "reg_weight": [1e-4],
            }
        )
        # 2*3*4*1 = 24
    elif model_name in ("HAGN", "HAGNUniform", "HAGNFixed", "HAGNGlobal"):
        # full target grid (36 trials) — restore when time allows:
        # hp_config['learning_rate'] = [5e-3, 1e-3, 5e-4, 1e-4]
        # hp_config['n_layers']      = [2, 3, 4]
        # hp_config['reg_weight']    = [1e-3, 1e-4, 1e-5]
        hp_config["learning_rate"] = [5e-3]
        hp_config["n_layers"] = [2, 3, 4]
        hp_config["reg_weight"] = [1e-3, 1e-4, 1e-5]
        # 1*3*3 = 9
    elif model_name in ("CSGCN", "CSGCNNoWarmup"):
        # lr=0.005 chosen in all real+synth runs
        hp_config["learning_rate"] = [0.005]
        hp_config["n_layers"] = [2, 3, 4]
        hp_config["reg_weight"] = [1e-3, 1e-4, 1e-5]
        if model_name == "CSGCNNoWarmup":
            hp_config["warmup_epochs"] = [0]
            hp_config["anneal_epochs"] = [20, 50]
        else:
            hp_config["warmup_epochs"] = [10, 30]
            hp_config["anneal_epochs"] = [20, 50]
        # CSGCN: 1*3*3*2*2 = 36, CSGCNNoWarmup: 1*3*3*1*2 = 18
    elif model_name in ("ASLGCN", "ASLGCNSymmetric"):
        # full target grid (108 trials) — restore when time allows:
        # hp_config['learning_rate'] = [5e-3, 1e-3, 5e-4, 1e-4]
        # hp_config['n_layers']      = [2, 3, 4]
        # hp_config['reg_weight']    = [1e-3, 1e-4, 1e-5]
        # hp_config['social_layers'] = [1, 2, 3]
        hp_config["learning_rate"] = [5e-3]
        hp_config["n_layers"] = [2, 3, 4]
        hp_config["reg_weight"] = [1e-3, 1e-4, 1e-5]
        hp_config["social_layers"] = [1, 2, 3]
        # 1*3*3*3 = 27

    hp_file = os.path.join(run_dir, "tuning.hyper")
    with open(hp_file, "w") as f:
        for k, v in hp_config.items():
            f.write(f"{k} choice {v}\n")

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

    if model_name in ("CSGCN", "CSGCNNoWarmup"):
        config_dict["warmup_epochs"] = 0 if model_name == "CSGCNNoWarmup" else 20
        config_dict["anneal_epochs"] = 30
    if model_name in ("ASLGCN", "ASLGCNSymmetric"):
        config_dict["social_layers"] = 1

    real_datasets = ["yelp", "douban-book", "lastfm"]

    # reduce search space for real datasets (much larger, so each trial takes 1-3h)
    if dataset_name in real_datasets:
        if model_name == "SimGCL":
            # based on lastfm+synth results: eps almost always 0.2, temp=0.2 on real,
            # lr always 0.001 or 0.0005, n_layers=2 on real, lambda varies
            hp_config["learning_rate"] = [0.001, 0.0005]
            hp_config["n_layers"] = [2]
            hp_config["temperature"] = [0.2]
            hp_config["lambda"] = [0.1, 0.5, 1.0, 2.0]
            hp_config["eps"] = [0.2]
            # 2*1*1*1*4*1 = 8
        elif model_name == "MHCN":
            # real best: lr in {0.001, 0.0001}, n_layers=3, ssl_reg in {1e-6, 5e-6}
            hp_config["learning_rate"] = [0.001, 0.0001]
            hp_config["n_layers"] = [2, 3]
            hp_config["ssl_reg"] = [5e-5, 1e-6, 1e-5]
            # 2*2*2*1 = 8
        elif model_name in ("CSGCN", "CSGCNNoWarmup"):
            # real best: lr=0.005 always, n_layers varies, warmup/anneal vary
            hp_config["learning_rate"] = [0.005]
            hp_config["n_layers"] = [2, 3, 4]
            hp_config["warmup_epochs"] = (
                [10, 30] if model_name != "CSGCNNoWarmup" else hp_config["warmup_epochs"]
            )
            hp_config["anneal_epochs"] = [20, 50]
            # 2*3*3*2*2 = 72
        # HAGN/ASLGCN families: no real-dataset override — use the unified grid above

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

        print("\nPhase 1: Tuning")
        # RecBole's default early_stop=10 halts after 10 trials with no improvement,
        # but exhaustive_search samples in random order so the default config may
        # never be reached — set early_stop >= space size to try all combos
        # (space size is counted below from hp_config)
        n_combos = 1
        for v in hp_config.values():
            n_combos *= len(v)
        print(f"  Search space: {n_combos} combinations")
        hp = HyperTuning(
            thesis_objective_function,
            algo="exhaustive",
            params_file=hp_file,
            fixed_config_file_list=[config_yaml],
            early_stop=max(n_combos, 100),
        )
        hp.run()

        print(f"\n✅ Best Params: {hp.best_params}")
        import logging

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        final_config_dict = config_dict.copy()
        final_config_dict.update(hp.best_params)
        final_config_dict["show_progress"] = True

        print("\nPhase 2: Final Run with Best Hyperparameters")

        gnn_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RecBole-GNN"
        )
        if gnn_path not in sys.path:
            sys.path.append(gnn_path)
        try:
            import recbole_gnn.model  # noqa: F401
        except ImportError:
            pass

        inference_time_s = None
        train_time_s = None
        n_params = None

        if model_name in ("CSGCN", "CSGCNNoWarmup"):
            from recbole_gnn.config import Config
            from recbole_gnn.model.social_recommender.csgcn import CSGCN, CSGCNTrainer
            from recbole_gnn.utils import create_dataset, data_preparation

            cfg = Config(model=CSGCN, dataset=dataset_name, config_dict=final_config_dict)
            ds = create_dataset(cfg)
            train_data, valid_data, test_data = data_preparation(cfg, ds)

            model = CSGCN(cfg, ds).to(cfg["device"])
            n_params = sum(p.numel() for p in model.parameters())
            trainer = CSGCNTrainer(cfg, model)
            _patch_trainer_for_epoch_tracking(trainer)

            t_train = time.time()
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=True
            )
            train_time_s = time.time() - t_train
            t0 = time.time()
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
            inference_time_s = time.time() - t0
            result = {"best_valid_result": best_valid_result, "test_result": test_result}

        elif model_name == "ASLGCN":
            from recbole_gnn.config import Config
            from recbole_gnn.model.social_recommender.aslgcn import ASLGCN
            from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

            cfg = Config(model=ASLGCN, dataset=dataset_name, config_dict=final_config_dict)
            ds = create_dataset(cfg)
            train_data, valid_data, test_data = data_preparation(cfg, ds)

            model = ASLGCN(cfg, ds).to(cfg["device"])
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

            try:
                import glob

                ckpt_files = sorted(
                    glob.glob(os.path.join(final_config_dict["checkpoint_dir"], "ASLGCN-*.pth"))
                )
                if ckpt_files:
                    m = ASLGCN(cfg, ds).to(cfg["device"])
                    ckpt = torch.load(ckpt_files[-1], map_location=cfg["device"])
                    m.load_state_dict(ckpt["state_dict"])
                    lam = m.get_lambda_social()
                    print(f"\n🔬 Learned λ_social = {lam:.4f} on {dataset_name}")
                    with open(os.path.join(run_dir, "lambda_social.txt"), "w") as f:
                        f.write(
                            f"dataset={dataset_name}\n"
                            f"lambda_social={lam:.6f}\n"
                            f"lambda_raw={m._lambda_raw.item():.6f}\n"
                        )
            except Exception as diag_e:
                print(f"[ASLGCN diagnostic] Could not extract lambda: {diag_e}")

        elif model_name in GNN_MODELS:
            from recbole_gnn.config import Config
            from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

            cfg = Config(model=model_name, dataset=dataset_name, config_dict=final_config_dict)
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
                model=model_name, dataset=dataset_name, config_dict=final_config_dict
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

        print(f"\n✅ Final Metrics: {result}")

        row_data = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Mode": "TUNED",
            "dataset_type": dataset_type,
            "seed": model_seed,
            "data_seed": data_seed,
            "total_epochs_trained": total_epochs,
            "best_epoch": best_epoch,
            "Best_Params": str(hp.best_params),
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

        with open(os.path.join(run_dir, "best_hyperparameters.yaml"), "w") as f:
            yaml.dump(hp.best_params, f)

        latex_vals = [
            f"{res_df.iloc[0][c]:.4f}"
            if isinstance(res_df.iloc[0].get(c), (float, np.float64))
            else str(res_df.iloc[0].get(c, "N/A"))
            for c in res_df.columns
        ]
        with open(os.path.join(run_dir, "latex_row.tex"), "w") as f:
            f.write(
                "% " + " & ".join(res_df.columns) + " \\\\\n" + " & ".join(latex_vals) + " \\\\\n"
            )

        print(f"✅ Success: {run_dir}")
        _cleanup_checkpoints(run_dir)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ FAILURE: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BPR")
    parser.add_argument("--dataset", type=str, default="yelp")
    parser.add_argument("--batch", type=str, default="")
    parser.add_argument("--seed", type=int, default=2020, help="Model/optimizer random seed")
    parser.add_argument(
        "--data_seed", type=int, default=1, help="Data seed (1=original ./datasets/)"
    )
    args, _ = parser.parse_known_args()
    run_pipeline(args.model, args.dataset, args.batch, args.seed, args.data_seed)
