#!/bin/bash
#SBATCH --job-name=cellot
#SBATCH --output log_cellot_train_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=syed.rizvi@yale.edu
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint="a100|a40"
#SBATCH --cpus-per-task=2
#SBATCH --mem=40gb
#SBATCH --time=2-00:00:00
date;hostname;pwd

module load miniconda
conda activate cellot
cd /home/sr2464/Desktop/cellot

export PYTHONPATH="/home/sr2464/Desktop/cellot"

# Runs
# All genes: /home/sr2464/scratch/C2S_Files/CellOT_files/pancreas_training_runs/run6_all_genes
# X_pca: /home/sr2464/scratch/C2S_Files/CellOT_files/pancreas_training_runs/run7_X_pca
# X_harmony: /home/sr2464/scratch/C2S_Files/CellOT_files/pancreas_training_runs/run8_X_harmony

python scripts/evaluate.py \
--outdir "/home/sr2464/scratch/C2S_Files/CellOT_files/pancreas_training_runs/run6_all_genes" \
--setting iid \
--where data_space