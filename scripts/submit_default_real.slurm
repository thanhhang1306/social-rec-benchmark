#!/bin/bash
#SBATCH --job-name=DefaultReal
#SBATCH --output=logs/DEFAULT_BATCH_%j_%a.log
#SBATCH --error=logs/DEFAULT_BATCH_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-23   # 3 datasets × 8 models = 24

BATCH_NAME="DEFAULT_BATCH_REAL"
DATA_SEED=2
MODEL_SEED=1

MODELS=("BPR" "LightGCN" "SimGCL" "DiffNet" "MHCN" "HAGN" "CSGCN" "ASLGCN")
DATASETS=(
    "yelp"
    "douban-book"
    "lastfm"
)

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 8))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 8))]}

if [ -z "$DATASET" ]; then
    echo "Out of range for task $SLURM_ARRAY_TASK_ID — skipping"
    exit 0
fi

scontrol update JobId=$SLURM_JOB_ID JobName="TR_${MODEL:0:4}_${DATASET}"

module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/recbench/new_rec/env
export PYTHONPATH=$PYTHONPATH:/n/fs/recbench/new_rec/RecBole-GNN
cd /n/fs/recbench/new_rec/

echo "Task $SLURM_ARRAY_TASK_ID → model=$MODEL  dataset=$DATASET"
python pipeline/run_default_pipeline.py --model=$MODEL --dataset=$DATASET --batch=$BATCH_NAME --seed=$MODEL_SEED --data_seed=$DATA_SEED