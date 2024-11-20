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

# Configs
# /home/sr2464/Desktop/cellot/configs/tasks/pancreas_cross_species.yaml
# /home/sr2464/Desktop/cellot/configs/tasks/pancreas_cross_species_X_pca.yaml
# /home/sr2464/Desktop/cellot/configs/tasks/pancreas_cross_species_X_harmony.yaml
# /home/sr2464/Desktop/cellot/configs/tasks/pancreas_cross_species_X_harmony_mouse_to_human.yaml

python scripts/train.py \
--outdir "/home/sr2464/scratch/C2S_Files/CellOT_files/pancreas_training_runs/run9_X_harmony_mouse_to_human" \
--config "/home/sr2464/Desktop/cellot/configs/tasks/pancreas_cross_species_X_harmony_mouse_to_human.yaml" \
--config "/home/sr2464/Desktop/cellot/configs/models/cellot.yaml"