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
# /home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/tasks/pancreas_cross_species_X_harmony.yaml
# /home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/tasks/pancreas_cross_species_X_harmony_mouse_to_human.yaml

python scripts/train.py \
--outdir "/home/dor3/cell2sentence/cross_species/CellOT_on_Homologous_Data/models/results_mouse_to_human2" \
--config "/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/tasks/pancreas_cross_species_X_harmony_mouse_to_human.yaml" \
--config "/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/models/cellot.yaml"



''' Legacy params: 
# Define paths to the configs and data
DATA_PATH="/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/datasets/harmony_common_pancreas_20241025.h5ad"
OUTDIR="/home/dor3/cell2sentence/cross_species/CellOT_on_Homologous_Data/results/mouse_to_human"
TASK_CONFIG="/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/tasks/pancreas_cross_species_X_harmony_mouse_to_human.yaml"
MODEL_CONFIG="/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/configs/models/cellot.yaml"
SCRIPTS_PATH="/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/scripts"

# Set species identifiers and other parameters
SOURCE_SPECIES="mouse"
TARGET_SPECIES="human"
BATCH_KEY="species_label"
EXP_GROUP="cross_species_cellot"
ONLINE="offline"
RESTART="false"
DEBUG="false"
DRY="false"
VERBOSE="true"

# Execute the CellOT training script with the specified parameters
python "/home/dor3/cell2sentence/cross_species/Syed_CellOT_Code/scripts/train.py" \
--config "${TASK_CONFIG}" \
--config "${MODEL_CONFIG}" \
--exp_group "${EXP_GROUP}" \
--online "${ONLINE}" \
--restart="${RESTART}" \
--debug="${DEBUG}" \
--dry="${DRY}" \
--verbose="${VERBOSE}"'''