#!/bin/bash
#SBATCH --job-name=TuneSynth
#SBATCH --output=logs/TUNED_SYNTH_%j_%a.log
#SBATCH --error=logs/TUNED_SYNTH_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=90:00:00
#SBATCH --array=0-999   # part A of 4224 (8 models × 176 datasets × 3 seed pairs); resubmit with OFFSET for B..E

# SLURM array max is 1000; resubmit with OFFSET to cover the full range:
#   part A: --array=0-999    OFFSET=0
#   part B: --array=0-999    OFFSET=1000
#   part C: --array=0-999    OFFSET=2000
#   part D: --array=0-999    OFFSET=3000
#   part E: --array=0-223    OFFSET=4000

BATCH_NAME="TUNED_BATCH_SYNTH"
OFFSET=${OFFSET:-0}
ACTUAL_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

MODELS=("BPR" "LightGCN" "SimGCL" "DiffNet" "MHCN" "HAGN" "CSGCN" "ASLGCN")

# dseed and mseed pair by combo index, not cartesian product
DATA_SEEDS=(1 2 3)
MODEL_SEEDS=(1 12 123)
N_COMBOS=3

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

N_MODELS=${#MODELS[@]}
N_DATASETS=${#DATASETS[@]}
JOBS_PER_COMBO=$((N_DATASETS * N_MODELS))

# ACTUAL_ID = combo_idx * JOBS_PER_COMBO + ds_idx * N_MODELS + model_idx
COMBO_IDX=$((ACTUAL_ID / JOBS_PER_COMBO))
REMAINDER=$((ACTUAL_ID % JOBS_PER_COMBO))
DS_IDX=$((REMAINDER / N_MODELS))
MODEL_IDX=$((REMAINDER % N_MODELS))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DS_IDX]}
DATA_SEED=${DATA_SEEDS[$COMBO_IDX]}
MODEL_SEED=${MODEL_SEEDS[$COMBO_IDX]}

if [ -z "$DATASET" ] || [ -z "$MODEL" ] || [ "$COMBO_IDX" -ge "$N_COMBOS" ]; then
    echo "Out of range for task $SLURM_ARRAY_TASK_ID — skipping"
    exit 0
fi

scontrol update JobId=$SLURM_JOB_ID JobName="TS_${MODEL:0:4}_${DATASET:0:20}_d${DATA_SEED}s${MODEL_SEED}"

module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/recbench/new_rec/env
export PYTHONPATH=$PYTHONPATH:/n/fs/recbench/new_rec/RecBole-GNN
cd /n/fs/recbench/new_rec/

echo "Task $SLURM_ARRAY_TASK_ID (ACTUAL_ID=$ACTUAL_ID) → model=$MODEL  dataset=$DATASET  dseed=$DATA_SEED  mseed=$MODEL_SEED"
python pipeline/run_tuned_pipeline.py --model=$MODEL --dataset=$DATASET --batch=$BATCH_NAME --seed=$MODEL_SEED --data_seed=$DATA_SEED
