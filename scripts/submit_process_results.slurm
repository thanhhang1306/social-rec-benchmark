#!/bin/bash
#SBATCH --job-name=ProcessResults
#SBATCH --output=logs/PROCESS_RESULTS_%j.log
#SBATCH --error=logs/PROCESS_RESULTS_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

module load anaconda3/2024.02
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/recbench/new_rec/env

cd /n/fs/recbench/new_rec/

python pipeline/process_results.py
python pipeline/process_aggregate.py
