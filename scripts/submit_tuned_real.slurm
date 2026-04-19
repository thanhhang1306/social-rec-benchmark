#!/bin/bash
#SBATCH --job-name=TuneReal
#SBATCH --output=logs/TUNED_REAL_%j_%a.log
#SBATCH --error=logs/TUNED_REAL_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --array=0-71   # 3 datasets × 3 seeds × 8 models = 72

BATCH_NAME="TUNED_BATCH_REAL"

MODELS=("BPR" "LightGCN" "SimGCL" "DiffNet" "MHCN" "HAGN" "CSGCN" "ASLGCN")
DATASETS=("yelp" "douban-book" "lastfm")
MODEL_SEEDS=(1 12 123)

N_MODELS=${#MODELS[@]}
N_DATASETS=${#DATASETS[@]}
N_SEEDS=${#MODEL_SEEDS[@]}

# TASK_ID = seed_idx * (N_DATASETS * N_MODELS) + ds_idx * N_MODELS + model_idx
SEED_IDX=$((SLURM_ARRAY_TASK_ID / (N_DATASETS * N_MODELS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (N_DATASETS * N_MODELS)))
DS_IDX=$((REMAINDER / N_MODELS))
MODEL_IDX=$((REMAINDER % N_MODELS))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DS_IDX]}
MODEL_SEED=${MODEL_SEEDS[$SEED_IDX]}

if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
    echo "Out of range for task $SLURM_ARRAY_TASK_ID — skipping"
    exit 0
fi

scontrol update JobId=$SLURM_JOB_ID JobName="TR_${MODEL:0:4}_${DATASET}_s${MODEL_SEED}"

module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/recbench/new_rec/env
export PYTHONPATH=$PYTHONPATH:/n/fs/recbench/new_rec/RecBole-GNN
cd /n/fs/recbench/new_rec/

echo "Task $SLURM_ARRAY_TASK_ID → model=$MODEL  dataset=$DATASET  mseed=$MODEL_SEED"
python pipeline/run_tuned_pipeline.py --model=$MODEL --dataset=$DATASET --batch=$BATCH_NAME --seed=$MODEL_SEED --data_seed=1
