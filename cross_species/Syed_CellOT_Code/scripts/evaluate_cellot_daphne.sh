#!/bin/bash
#SBATCH --job-name=cellot_cross_species
#SBATCH --output log_cellot_train_%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daphne.raskin@yale.edu
#SBATCH --partition bigmem
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150gb
#SBATCH --time=24:00:00

# Print date, hostname, and current working directory
date;hostname;pwd

# Load Miniconda and activate correct environment
module load miniconda
conda activate cellot2
cd /home/dor3/cell2sentence/cross_species/Syed_CellOT_Code

# Set PYTHONPATH to ensure CellOT can be imported
export PYTHONPATH="/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code"

# Configs (run 2 tasks separately on 2 instances of same model)
# human -> mouse:
# /home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/tasks/pancreas_cross_species_X_harmony.yaml
# mouse -> human:
# /home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/tasks/pancreas_cross_species_X_harmony_mouse_to_human.yaml

python scripts/evaluate.py \
--outdir "/home/dor3/cell2sentence/cross_species/CellOT_on_Homologous_Data/results_mouse_to_human" \
--setting iid \
--where data_space