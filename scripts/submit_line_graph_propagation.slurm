#!/bin/bash
#SBATCH --job-name=LineGraphProp
#SBATCH --output=logs/LINE_PROP_%j_%a.log
#SBATCH --error=logs/LINE_PROP_%j_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-152   # 17 datasets × 9 seed combos = 153

DATASETS=(
    "line_graph_u2000_i10000_sp0.002_ns0.2_max30_decay0.5"
    "line_graph_u2000_i10000_sp0.01_ns0.1_max30_decay0.5"
    "line_graph_u2000_i2000_sp0.01_ns0.1_max30_decay0.5"
    "line_graph_u2000_i2000_sp0.02_ns0.1_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.003_ns0.1_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.005_ns0.1_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.01_ns0.02_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.01_ns0.05_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.01_ns0.1_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.01_ns0.2_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.01_ns0.3_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.01_ns0.5_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.02_ns0.1_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.02_ns0.2_max30_decay0.5"
    "line_graph_u2000_i5000_sp0.05_ns0.1_max30_decay0.5"
    "line_graph_u5000_i5000_sp0.01_ns0.1_max30_decay0.5"
    "line_graph_u500_i5000_sp0.01_ns0.1_max30_decay0.5"
)

DATA_SEEDS=(1 2 3)
MODEL_SEEDS=(1 12 123)

N_DATASETS=${#DATASETS[@]}
N_COMBOS=9

DS_IDX=$((SLURM_ARRAY_TASK_ID / N_COMBOS))
COMBO_IDX=$((SLURM_ARRAY_TASK_ID % N_COMBOS))
DSEED_IDX=$((COMBO_IDX / 3))
MSEED_IDX=$((COMBO_IDX % 3))

DATASET=${DATASETS[$DS_IDX]}
DSEED=${DATA_SEEDS[$DSEED_IDX]}
MSEED=${MODEL_SEEDS[$MSEED_IDX]}

if [ -z "$DATASET" ]; then
    echo "Out of range for task $SLURM_ARRAY_TASK_ID — skipping"
    exit 0
fi

scontrol update JobId=$SLURM_JOB_ID JobName="LP_${DATASET:11:15}_d${DSEED}s${MSEED}"

module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/recbench/new_rec/env
export PYTHONPATH=$PYTHONPATH:/n/fs/recbench/new_rec/RecBole-GNN
cd /n/fs/recbench/new_rec/

echo "Task $SLURM_ARRAY_TASK_ID → dataset=$DATASET dseed=$DSEED mseed=$MSEED"
python -u pipeline/line_graph_propagation.py \
    --dataset $DATASET --dseed $DSEED --mseed $MSEED \
    --bins 10 20 50
