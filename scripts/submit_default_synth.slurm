#!/bin/bash
#SBATCH --job-name=DefaultSynth
#SBATCH --output=logs/DEFAULT_BATCH_%j_%a.log
#SBATCH --error=logs/DEFAULT_BATCH_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-999   # part A of 1408 (176 datasets × 8 models); resubmit --array=1000-1407 for B

BATCH_NAME="DEFAULT_BATCH_SYNTH"
DATA_SEED=3
MODEL_SEED=123

MODELS=("BPR" "LightGCN" "SimGCL" "DiffNet" "MHCN" "HAGN" "CSGCN" "ASLGCN")
DATASETS=(
    "contrarian_u2000_i10000_sp0.002_ns0.2"
    "echo_chamber_u2000_i10000_sp0.002_ns0.2"
    "fake_users_u2000_i10000_sp0.002_ns0.2_fake0.1"
    "line_graph_u2000_i10000_sp0.002_ns0.2_max30_decay0.5"
    "partial_alignment_u2000_i10000_sp0.002_ns0.2"
    "random_u2000_i10000_sp0.002_ns0.2"
    "scale_free_u2000_i10000_sp0.002_ns0.2"
    "small_world_u2000_i10000_sp0.002_ns0.2"
    "star_u2000_i10000_sp0.002_ns0.2"

    "contrarian_u2000_i5000_sp0.01_ns0.2"
    "echo_chamber_u2000_i5000_sp0.01_ns0.2"
    "fake_users_u2000_i5000_sp0.01_ns0.2_fake0.1"
    "line_graph_u2000_i5000_sp0.01_ns0.2_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.01_ns0.2"
    "random_u2000_i5000_sp0.01_ns0.2"
    "scale_free_u2000_i5000_sp0.01_ns0.2"
    "small_world_u2000_i5000_sp0.01_ns0.2"
    "star_u2000_i5000_sp0.01_ns0.2"

    "contrarian_u2000_i5000_sp0.01_ns0.1"
    "echo_chamber_u2000_i5000_sp0.01_ns0.1"
    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.1"
    "line_graph_u2000_i5000_sp0.01_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1"
    "random_u2000_i5000_sp0.01_ns0.1"
    "scale_free_u2000_i5000_sp0.01_ns0.1"
    "small_world_u2000_i5000_sp0.01_ns0.1"
    "star_u2000_i5000_sp0.01_ns0.1"

    "contrarian_u2000_i5000_sp0.02_ns0.1"
    "echo_chamber_u2000_i5000_sp0.02_ns0.1"
    "fake_users_u2000_i5000_sp0.02_ns0.1_fake0.1"
    "line_graph_u2000_i5000_sp0.02_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.02_ns0.1"
    "random_u2000_i5000_sp0.02_ns0.1"
    "scale_free_u2000_i5000_sp0.02_ns0.1"
    "small_world_u2000_i5000_sp0.02_ns0.1"
    "star_u2000_i5000_sp0.02_ns0.1"

    "contrarian_u2000_i5000_sp0.005_ns0.1"
    "echo_chamber_u2000_i5000_sp0.005_ns0.1"
    "fake_users_u2000_i5000_sp0.005_ns0.1_fake0.1"
    "line_graph_u2000_i5000_sp0.005_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.005_ns0.1"
    "random_u2000_i5000_sp0.005_ns0.1"
    "scale_free_u2000_i5000_sp0.005_ns0.1"
    "small_world_u2000_i5000_sp0.005_ns0.1"
    "star_u2000_i5000_sp0.005_ns0.1"

    "contrarian_u2000_i5000_sp0.02_ns0.2"
    "echo_chamber_u2000_i5000_sp0.02_ns0.2"
    "fake_users_u2000_i5000_sp0.02_ns0.2_fake0.1"
    "line_graph_u2000_i5000_sp0.02_ns0.2_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.02_ns0.2"
    "random_u2000_i5000_sp0.02_ns0.2"
    "scale_free_u2000_i5000_sp0.02_ns0.2"
    "small_world_u2000_i5000_sp0.02_ns0.2"
    "star_u2000_i5000_sp0.02_ns0.2"

    "contrarian_u2000_i5000_sp0.01_ns0.05"
    "echo_chamber_u2000_i5000_sp0.01_ns0.05"
    "fake_users_u2000_i5000_sp0.01_ns0.05_fake0.1"
    "line_graph_u2000_i5000_sp0.01_ns0.05_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.01_ns0.05"
    "random_u2000_i5000_sp0.01_ns0.05"
    "scale_free_u2000_i5000_sp0.01_ns0.05"
    "small_world_u2000_i5000_sp0.01_ns0.05"
    "star_u2000_i5000_sp0.01_ns0.05"

    "contrarian_u2000_i2000_sp0.01_ns0.1"
    "echo_chamber_u2000_i2000_sp0.01_ns0.1"
    "fake_users_u2000_i2000_sp0.01_ns0.1_fake0.1"
    "line_graph_u2000_i2000_sp0.01_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i2000_sp0.01_ns0.1"
    "random_u2000_i2000_sp0.01_ns0.1"
    "scale_free_u2000_i2000_sp0.01_ns0.1"
    "small_world_u2000_i2000_sp0.01_ns0.1"
    "star_u2000_i2000_sp0.01_ns0.1"

    "contrarian_u2000_i2000_sp0.02_ns0.1"
    "echo_chamber_u2000_i2000_sp0.02_ns0.1"
    "fake_users_u2000_i2000_sp0.02_ns0.1_fake0.1"
    "line_graph_u2000_i2000_sp0.02_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i2000_sp0.02_ns0.1"
    "random_u2000_i2000_sp0.02_ns0.1"
    "scale_free_u2000_i2000_sp0.02_ns0.1"
    "small_world_u2000_i2000_sp0.02_ns0.1"
    "star_u2000_i2000_sp0.02_ns0.1"

    "contrarian_u2000_i10000_sp0.01_ns0.1"
    "echo_chamber_u2000_i10000_sp0.01_ns0.1"
    "fake_users_u2000_i10000_sp0.01_ns0.1_fake0.1"
    "line_graph_u2000_i10000_sp0.01_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i10000_sp0.01_ns0.1"
    "random_u2000_i10000_sp0.01_ns0.1"
    "scale_free_u2000_i10000_sp0.01_ns0.1"
    "small_world_u2000_i10000_sp0.01_ns0.1"
    "star_u2000_i10000_sp0.01_ns0.1"

    "contrarian_u2000_i5000_sp0.01_ns0.3"
    "echo_chamber_u2000_i5000_sp0.01_ns0.3"
    "fake_users_u2000_i5000_sp0.01_ns0.3_fake0.1"
    "line_graph_u2000_i5000_sp0.01_ns0.3_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.01_ns0.3"
    "random_u2000_i5000_sp0.01_ns0.3"
    "scale_free_u2000_i5000_sp0.01_ns0.3"
    "small_world_u2000_i5000_sp0.01_ns0.3"
    "star_u2000_i5000_sp0.01_ns0.3"

    "contrarian_u2000_i5000_sp0.01_ns0.5"
    "echo_chamber_u2000_i5000_sp0.01_ns0.5"
    "fake_users_u2000_i5000_sp0.01_ns0.5_fake0.1"
    "line_graph_u2000_i5000_sp0.01_ns0.5_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.01_ns0.5"
    "random_u2000_i5000_sp0.01_ns0.5"
    "scale_free_u2000_i5000_sp0.01_ns0.5"
    "small_world_u2000_i5000_sp0.01_ns0.5"
    "star_u2000_i5000_sp0.01_ns0.5"

    "contrarian_u2000_i5000_sp0.01_ns0.02"
    "echo_chamber_u2000_i5000_sp0.01_ns0.02"
    "fake_users_u2000_i5000_sp0.01_ns0.02_fake0.1"
    "line_graph_u2000_i5000_sp0.01_ns0.02_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.01_ns0.02"
    "random_u2000_i5000_sp0.01_ns0.02"
    "scale_free_u2000_i5000_sp0.01_ns0.02"
    "small_world_u2000_i5000_sp0.01_ns0.02"
    "star_u2000_i5000_sp0.01_ns0.02"

    "contrarian_u2000_i5000_sp0.003_ns0.1"
    "echo_chamber_u2000_i5000_sp0.003_ns0.1"
    "fake_users_u2000_i5000_sp0.003_ns0.1_fake0.1"
    "line_graph_u2000_i5000_sp0.003_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.003_ns0.1"
    "random_u2000_i5000_sp0.003_ns0.1"
    "scale_free_u2000_i5000_sp0.003_ns0.1"
    "small_world_u2000_i5000_sp0.003_ns0.1"
    "star_u2000_i5000_sp0.003_ns0.1"

    "contrarian_u2000_i5000_sp0.05_ns0.1"
    "echo_chamber_u2000_i5000_sp0.05_ns0.1"
    "fake_users_u2000_i5000_sp0.05_ns0.1_fake0.1"
    "line_graph_u2000_i5000_sp0.05_ns0.1_max30_decay0.5"
    "partial_alignment_u2000_i5000_sp0.05_ns0.1"
    "random_u2000_i5000_sp0.05_ns0.1"
    "scale_free_u2000_i5000_sp0.05_ns0.1"
    "small_world_u2000_i5000_sp0.05_ns0.1"
    "star_u2000_i5000_sp0.05_ns0.1"

    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.05"
    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.2"
    "fake_users_u2000_i5000_sp0.01_ns0.1_fake0.4"

    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd1"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd3"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd7"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_sd9"

    "random_u2000_i5000_sp0.01_ns0.1_erp0.001"
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg2.0"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg2.0"
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg2.0"

    "random_u2000_i5000_sp0.01_ns0.1_erp0.002"
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg4.0"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg4.0"
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg4.0"

    "random_u2000_i5000_sp0.01_ns0.1_erp0.005"
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg10.0"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg10.0"
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg10.0"

    "random_u2000_i5000_sp0.01_ns0.1_erp0.01"
    "echo_chamber_u2000_i5000_sp0.01_ns0.1_deg20.0"
    "partial_alignment_u2000_i5000_sp0.01_ns0.1_deg20.0"
    "contrarian_u2000_i5000_sp0.01_ns0.1_deg20.0"

    "contrarian_u500_i5000_sp0.01_ns0.1"
    "echo_chamber_u500_i5000_sp0.01_ns0.1"
    "fake_users_u500_i5000_sp0.01_ns0.1_fake0.1"
    "line_graph_u500_i5000_sp0.01_ns0.1_max30_decay0.5"
    "partial_alignment_u500_i5000_sp0.01_ns0.1"
    "random_u500_i5000_sp0.01_ns0.1"
    "scale_free_u500_i5000_sp0.01_ns0.1"
    "small_world_u500_i5000_sp0.01_ns0.1"
    "star_u500_i5000_sp0.01_ns0.1"

    "contrarian_u5000_i5000_sp0.01_ns0.1"
    "echo_chamber_u5000_i5000_sp0.01_ns0.1"
    "fake_users_u5000_i5000_sp0.01_ns0.1_fake0.1"
    "line_graph_u5000_i5000_sp0.01_ns0.1_max30_decay0.5"
    "partial_alignment_u5000_i5000_sp0.01_ns0.1"
    "random_u5000_i5000_sp0.01_ns0.1"
    "scale_free_u5000_i5000_sp0.01_ns0.1"
    "small_world_u5000_i5000_sp0.01_ns0.1"
    "star_u5000_i5000_sp0.01_ns0.1"
)

N_DATASETS=${#DATASETS[@]}

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 8))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 8))]}

if [ -z "$DATASET" ]; then
    echo "Out of range for task $SLURM_ARRAY_TASK_ID — skipping"
    exit 0
fi

scontrol update JobId=$SLURM_JOB_ID JobName="TS_${MODEL:0:4}_${DATASET:0:30}"

module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/recbench/new_rec/env
export PYTHONPATH=$PYTHONPATH:/n/fs/recbench/new_rec/RecBole-GNN
cd /n/fs/recbench/new_rec/

echo "Task $SLURM_ARRAY_TASK_ID → model=$MODEL  dataset=$DATASET"
python pipeline/run_default_pipeline.py --model=$MODEL --dataset=$DATASET --batch=$BATCH_NAME --seed=$MODEL_SEED --data_seed=$DATA_SEED