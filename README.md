# Social Recommendation Benchmark

A benchmarking framework for social recommendation models on both real-world and synthetic graph datasets. Evaluates how different social network topologies affect recommendation quality and how well models exploit social signals.

---

## Overview

**Models evaluated:**

| Type | Models |
|---|---|
| CF baselines | BPR, LightGCN, SimGCL |
| Social GNNs | DiffNet, MHCN |
| Custom social GNNs | HAGN, CSGCN, ASLGCN |
| Ablation variants | HAGNUniform, HAGNFixed, CSGCNNoWarmup, ASLGCNSymmetric |

**Datasets:**

| Type | Datasets |
|---|---|
| Real | Yelp, Douban-Book, LastFM |
| Synthetic | 176 graphs across 9 topologies × parameter sweeps, generated per data seed |

Synthetic topologies: `contrarian`, `echo_chamber`, `partial_alignment`, `scale_free`, `small_world`, `random`, `star`, `line_graph`, `fake_users`.

**Experiment structure:**
- **Data seeds** (`dseed_1/2/3`): independent graph realizations for synthetic datasets
- **Model seeds** (`mseed_1/12/123`): independent training runs to measure variance
- **Default batch**: models run with fixed hyperparameters
- **Tuned batch**: exhaustive hyperparameter search via RecBole HyperTuning
- **Ablation batch**: ablation variants trained with the default hyperparameters of their parent model

---

## Setup

### Create environment

The pinned package set in `requirements.txt` targets Python 3.10, CUDA 11.8, and PyTorch 2.0.1 (to match the `torch-*` geometric wheels). A conda env at a dedicated path is recommended so that the Slurm scripts can activate it directly:

```bash
conda create -p /path/to/env python=3.10
conda activate /path/to/env
pip install -r requirements.txt
```

### Point the Slurm scripts at your environment

Every script in `scripts/` activates the conda env via an absolute path. Open each `.slurm` file and replace the placeholder path with yours:

```bash
conda activate /n/fs/recbench/new_rec/env   # ← replace with your env path
export PYTHONPATH=$PYTHONPATH:/n/fs/recbench/new_rec/RecBole-GNN
cd /n/fs/recbench/new_rec/                  # ← replace with your repo path
```

The `PYTHONPATH` export adds the vendored RecBole-GNN fork (which includes the custom models) to the import path.

### Real datasets

Place the real datasets under `datasets/` (tab-separated RecBole token format):

```
datasets/
  yelp/           yelp.inter           yelp.net
  douban-book/    douban-book.inter    douban-book.net
  lastfm/         lastfm.inter         lastfm.net
```

Each `.inter` file uses columns `user_id:token`, `item_id:token` (plus optional `rating:float`). Each `.net` file contains directed social edges as `source_id:token`, `target_id:token`.

### Synthetic datasets

Generate all 176 synthetic datasets for a given data seed:

```bash
bash generate/run_generation.bash
```

Generated datasets land in `datasets/dseed_N/`.

### Verify setup

```bash
python smoke_test.py
```

Runs all 8 models for 2 epochs on a small dataset to confirm imports and pipeline logic before submitting full jobs.

---

## Running experiments

### Single run (local / interactive)

```bash
# default (untuned) — real dataset
python pipeline/run_default_pipeline.py --model LightGCN --dataset yelp \
    --batch DEFAULT_BATCH_REAL --seed 1

# default — synthetic dataset (requires --data_seed)
python pipeline/run_default_pipeline.py --model MHCN \
    --dataset contrarian_u2000_i5000_sp0.01_ns0.1 \
    --batch DEFAULT_BATCH_SYNTH --seed 1 --data_seed 2

# tuned (hyperparameter search)
python pipeline/run_tuned_pipeline.py --model LightGCN --dataset yelp \
    --batch TUNED_BATCH_REAL --seed 1
```

Results are saved under `results/{batch}/mseed_{seed}/{dataset}/{model}/` (real) or `results/{batch}/dseed_{data_seed}/mseed_{seed}/{dataset}/{model}/` (synthetic).

### Slurm cluster

Submit full batches using the scripts in `scripts/`:

```bash
# default (untuned) hyperparameters
sbatch scripts/submit_default_real.slurm
sbatch scripts/submit_default_synth.slurm

# tuned (hyperparameter search)
sbatch scripts/submit_tuned_real.slurm
sbatch scripts/submit_tuned_synth.slurm

# ablation variants
sbatch scripts/submit_ablation_real.slurm
sbatch scripts/submit_ablation_synth.slurm

# post-processing: parse Slurm logs into per-run CSVs and build results/all_results.csv
sbatch scripts/submit_process_results.slurm

# line graph per-user propagation analysis (thesis §4.4)
sbatch scripts/submit_line_graph_propagation.slurm
```

The tuned-synth batch has 4224 tasks (8 models × 176 datasets × 3 seed pairs) and exceeds Slurm's 1000-task array limit — resubmit with the `OFFSET` env variable as documented at the top of `submit_tuned_synth.slurm`. All jobs write logs to `logs/`.

---

## Repository layout

```
new_rec/
  analysis/     run-status checks, thesis figures, exploratory plots
  datasets/     real datasets + dseed_N/ synthetic datasets
  exploration/  outputs of analysis/exploration/
  figures/      outputs of analysis/thesis/ (publication figures)
  generate/     synthetic dataset generators
  pipeline/     training and post-processing entry points
  RecBole-GNN/  vendored RecBole-GNN fork, extended with custom models
  results/      experiment outputs (final_metrics.csv, epoch_data.csv, configs)
  scripts/      Slurm job scripts (8 total)
  smoke_test.py
  requirements.txt
  pyproject.toml
```

### `analysis/`

Three subfolders grouped by purpose:

- **`checks/`** — quick scripts for tracking completion status across batches.
  - `check_exist.py` — count done / missing / failed per (batch, model, dataset).
  - `check_finish.py` — count runs whose `final_metrics.csv` and `epoch_data.csv` both exist.
  - `check_names.py` — verify that all expected synthetic dataset folders are present.
  - `check_ablations.py` — completion status specifically for ablation variants.
- **`thesis/`** — one script per thesis figure or statistical table, reading from `results/all_results.csv`. Outputs go to `figures/`.
  - Methodology (Ch. 3): `thesis_methodology.py`, `topology_diagrams.py`.
  - Primary results (§4.3): `primary_heatmap.py`, `structural_sweep.py`, `structural_sweep_stats.py`, `appendix_structural.py`.
  - Shared sweeps and line graph (§4.4): `shared_sweep.py`, `line_graph_density.py`.
  - Tuning (§4.7): `tuning_crossings.py`, `tuning_synth_stats.py`, `best_config_modes_table.py`.
  - Hybrids and ablations (Ch. 5): `hybrid.py`.
  - Deployment (Ch. 6): `deployment.py`.
  - `thesis_palette.py` defines the shared color palette used across thesis figures.
- **`exploration/`** — general-purpose plotting for data and result exploration. Outputs go to `exploration/`.
  - `visualize_datasets.py` — per-dataset interaction and social-graph plots.
  - `visualize_compare.py` — cross-dataset comparison and multi-seed reproducibility.
  - `visualize_results.py` — aggregate model performance across batches.
  - `visualize_ablations.py` — ablation variants vs. their parent models.

### `datasets/`

Holds every dataset RecBole reads from. Real datasets sit at the top level (`yelp/`, `douban-book/`, `lastfm/`); synthetic datasets are grouped under per-seed directories (`dseed_1/`, `dseed_2/`, `dseed_3/`) so the same topology and parameter set appears in multiple realizations.

### `exploration/`

Output directory for `analysis/exploration/` scripts. Per-dataset plots sit under `exploration/dseed_N/<dataset>/`, cross-dataset comparisons under `exploration/dataset_comparison/`, and ablation summaries under `exploration/ablations/`.

### `figures/`

Output directory for `analysis/thesis/` scripts. Figures are organized by chapter:

```
figures/
  background/    topology schematics and background diagrams
  methodology/   dataset-generation figures
  results/       primary results and structural sweeps
  hybrids/       ablation and hybrid-model figures
  deployment/    deployment-chapter figures
  appendix/      supplementary figures
```

### `generate/`

Synthetic dataset generators. Each script takes `--topology`, `--n_users`, `--n_items`, `--sparsity`, `--noise`, and a `--seed`, and writes matching `.inter` and `.net` files into `./datasets/`.

- `generate_model.py` — Group I (no content-structure coupling): scale-free (BA), small-world (WS), random (ER).
- `generate_social.py` — Group II (community-based cosine structure): echo chamber, partial alignment, contrarian.
- `generate_topology.py` — Group III (special structures): star, line graph, fake users.
- `extract_diagnostics.py` — regenerate interest vectors for existing datasets and write per-dataset `diagnostics.json` files containing cosine and Jaccard signal statistics.
- `run_generation.bash` — drives all three generators across the full parameter grid for one data seed.

### `pipeline/`

Training and post-processing entry points.

- `run_default_pipeline.py` — trains one (model, dataset, seed) with fixed default hyperparameters.
- `run_tuned_pipeline.py` — runs RecBole's HyperTuning search over a curated grid per model, then trains the best configuration.
- `process_results.py` — parses per-run RecBole / Slurm logs into `epoch_data.csv` and `final_metrics.csv`.
- `process_aggregate.py` — concatenates every `final_metrics.csv` under `results/` into `results/all_results.csv`.
- `line_graph_propagation.py` — computes per-user performance vs. chain position on the line-graph topology (thesis §4.4).

### `RecBole-GNN/`

A vendored fork of [RecBole-GNN](https://github.com/RUCAIBox/RecBole-GNN). The upstream framework is preserved almost verbatim; the project adds:

- `recbole_gnn/model/social_recommender/hagn.py` — **HAGN**: LightGCN backbone with Jaccard-weighted social aggregation gated by a per-user learned scalar α_u. The gate MLP takes mean neighbor Jaccard, log interaction count, and log social degree as features. Ablations `HAGNUniform` (uniform edge weights) and `HAGNFixed` (α=1) live in the same file.
- `recbole_gnn/model/social_recommender/csgcn.py` — **CSGCN**: same backbone with a global curriculum weight β(t) that linearly introduces the social signal from 0 to 1 over training. Requires the custom `CSGCNTrainer`. Ablation `CSGCNNoWarmup` disables the curriculum.
- `recbole_gnn/model/social_recommender/aslgcn.py` — **ASLGCN**: asymmetric social propagation channel with Jaccard-weighted normalization and a single learned global scalar λ_social (softplus-constrained) gating the social contribution. Ablation `ASLGCNSymmetric` uses symmetric normalization.
- `recbole_gnn/properties/model/{HAGN,CSGCN,ASLGCN}.yaml` — default hyperparameters for each custom model.
- Minor edits to `recbole_gnn/quick_start.py` and `recbole_gnn/utils.py` to register the new models.

### `results/`

Experiment outputs organized by batch. Real batches key on `mseed_{N}/{dataset}/{model}/`; synthetic batches add a `dseed_{N}/` level above that. Each leaf contains at least:

- `final_metrics.csv` — final test metrics.
- `epoch_data.csv` — per-epoch training curves.
- `config.yaml` — the full resolved RecBole config.
- `best_hyperparameters.yaml` (tuned batches only) — the winning configuration from the hyperparameter search.

`results/all_results.csv` is the flat join of every `final_metrics.csv` under the tree; every thesis figure script consumes this file.

### `scripts/`

Slurm submission scripts (one per batch). They all share the same activation block and differ in array size, dataset list, and which pipeline entry point they invoke. Remember to update the env and repo paths in each file before submitting.

---

## Analysis workflows

Run from the project root.

### Check run status

```bash
python analysis/checks/check_exist.py
python analysis/checks/check_finish.py
python analysis/checks/check_names.py
python analysis/checks/check_ablations.py
```

### Process results

```bash
# parse per-run logs into epoch_data.csv + final_metrics.csv
python pipeline/process_results.py

# collect every final_metrics.csv into results/all_results.csv
python pipeline/process_aggregate.py
```

### Thesis figures

```bash
python analysis/thesis/thesis_methodology.py
python analysis/thesis/primary_heatmap.py
python analysis/thesis/structural_sweep.py
# ... etc. (see analysis/thesis/ for the full list)
```

### Data exploration

```bash
# per-dataset interaction and social-graph plots
python analysis/exploration/visualize_datasets.py --dseed 1

# cross-dataset comparison across data seeds
python analysis/exploration/visualize_compare.py \
    --stats exploration/dseed_1/dataset_stats.csv \
            exploration/dseed_2/dataset_stats.csv \
            exploration/dseed_3/dataset_stats.csv

# aggregate model-performance plots across batches
python analysis/exploration/visualize_results.py \
    --results_dir results/TUNED_BATCH_REAL results/TUNED_BATCH_SYNTH

# ablation variants vs. parent models
python analysis/exploration/visualize_ablations.py
```

---

## Acknowledgement

AI-assisted tools (Claude and ChatGPT) were used during this work for coding assistance, debugging, and generating diagrams. All experimental design, analysis, interpretation, and writing are my own.

