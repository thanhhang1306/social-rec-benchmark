"""Run all 8 models for 2 epochs on a small dataset to verify imports and pipeline."""

import argparse
import os
import sys
import time

import numpy as np
import yaml

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "bool_"):
    np.bool_ = np.bool8

gnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RecBole-GNN")
if gnn_path not in sys.path:
    sys.path.append(gnn_path)

import scipy.sparse

if not hasattr(scipy.sparse.dok_matrix, "_update"):
    scipy.sparse.dok_matrix._update = scipy.sparse.dok_matrix.update

MODELS = ["BPR", "LightGCN", "SimGCL", "DiffNet", "MHCN", "HAGN", "CSGCN", "ASLGCN"]
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

SMOKE_OVERRIDES = {
    "epochs": 2,
    "stopping_step": 2,
    "eval_step": 1,
    "train_batch_size": 512,
    "eval_batch_size": 4096000,
    "embedding_size": 16,
    "n_layers": 1,
}

real_datasets = ["yelp", "douban-book", "lastfm"]


def smoke_one(model_name, dataset_name, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    config_dict = {
        "data_path": "./datasets",
        "checkpoint_dir": os.path.join(run_dir, "checkpoints"),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": None,
        "net_file_extension": ".net",
        "social_id_field": "source_id",
        "fields_in_same_space": [["user_id", "source_id", "target_id"]],
        "load_col": {"inter": ["user_id", "item_id"], "net": ["source_id", "target_id"]},
        "user_inter_num_interval": "[10,inf)" if dataset_name in real_datasets else "[3,inf)",
        "item_inter_num_interval": "[10,inf)" if dataset_name in real_datasets else "[1,inf)",
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": {"valid": "full", "test": "full"},
        },
        "valid_metric": "NDCG@10",
        "metrics": ["Recall", "NDCG"],
        "topk": [10],
        "seed": 2020,
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1, "dynamic": False},
        **SMOKE_OVERRIDES,
    }

    # model-specific required keys
    if model_name in ("HAGN", "HAGNUniform", "HAGNFixed", "HAGNGlobal"):
        config_dict["require_pow"] = False
    elif model_name in ("CSGCN", "CSGCNNoWarmup"):
        config_dict["warmup_epochs"] = 0 if model_name == "CSGCNNoWarmup" else 1
        config_dict["anneal_epochs"] = 1
        config_dict["require_pow"] = False
    elif model_name in ("ASLGCN", "ASLGCNSymmetric"):
        config_dict["social_layers"] = 1
        config_dict["require_pow"] = False
    elif model_name == "SimGCL":
        config_dict["temperature"] = 0.1
        config_dict["lambda"] = 0.1
        config_dict["eps"] = 0.1

    config_yaml = os.path.join(run_dir, "smoke_config.yaml")
    with open(config_yaml, "w") as f:
        yaml.dump(config_dict, f)

    sys.argv = [sys.argv[0], f"--model={model_name}", f"--dataset={dataset_name}"]
    t_start = time.time()

    try:
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
            trainer = Trainer(cfg, model)

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
            trainer = CSGCNTrainer(cfg, model)

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
            trainer = Trainer(cfg, model)

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
            trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

        else:  # BPR
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
            trainer = get_recbole_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

        trainer.fit(train_data, valid_data, saved=False, show_progress=False)
        t0 = time.time()
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        inference_time_s = time.time() - t0
        elapsed = time.time() - t_start
        ndcg = test_result.get("ndcg@10", "N/A")
        print(
            f"  ✅ {model_name:<10} NDCG@10={ndcg:.4f}  inference={inference_time_s:.2f}s  total={elapsed:.1f}s"
        )
        return True

    except Exception as e:
        import traceback

        elapsed = time.time() - t_start
        print(f"  ❌ {model_name:<10} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="random_u500_i5000_sp0.01_ns0.1",
        help="Dataset to smoke-test against (use a small one)",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS, help="Subset of models to test (default: all 8)"
    )
    args = parser.parse_args()

    smoke_dir = "/tmp/smoke_test_runs"
    print(f"\nSmoke test: {args.models} on [{args.dataset}]\n")

    results = {}
    for m in args.models:
        run_dir = os.path.join(smoke_dir, args.dataset, m)
        results[m] = smoke_one(m, args.dataset, run_dir)

    passed = [m for m, ok in results.items() if ok]
    failed = [m for m, ok in results.items() if not ok]
    print(f"\nResults: {len(passed)}/{len(args.models)} passed")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)
