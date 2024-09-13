#!/bin/bash
#SBATCH --job-name="Loop-GSLAM"
#SBATCH --output=/home/stud/lxuh/storage/user/lxuh/output/logs/%A_%a.log  # please change accordingly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=16G
#SBATCH --partition=NORMAL
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=0-0 # number of scenes, 0-7 for Replica, 0-2 for TUM_RGBD, 0-5 for ScanNet, 0-4 for ScanNet++

dataset="TUM_RGBD" # set dataset
if [ "$dataset" == "Replica" ]; then
    scenes=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")
    INPUT_PATH="/home/stud/lxuh/storage/user/lxuh/data/Replica"
elif [ "$dataset" == "TUM_RGBD" ]; then
    # scenes=("rgbd_dataset_freiburg1_desk" "rgbd_dataset_freiburg2_xyz" "rgbd_dataset_freiburg3_long_office_household")
    # scenes=("rgbd_dataset_freiburg1_desk" "rgbd_dataset_freiburg3_long_office_household")
    scenes=("rgbd_dataset_freiburg3_long_office_household")
    INPUT_PATH="/home/stud/lxuh/storage/group/dataset_mirrors/tum_rgbd_benchmark"
elif [ "$dataset" == "ScanNet" ]; then
    # scenes=("scene0000_00" "scene0059_00" "scene0106_00" "scene0169_00" "scene0181_00" "scene0207_00")
    scenes=("scene0059_00" "scene0106_00" "scene0169_00" "scene0207_00")
    # scenes=("scene0059_00")
    INPUT_PATH="/home/stud/lxuh/storage/user/lxuh/data/ScanNet"
elif [ "$dataset" == "ScanNetPP" ]; then
    scenes=("b20a261fdf" "8b5caf3398" "fb05e13ad1" "2e74812d00" "281bc17764")
    INPUT_PATH="/home/stud/lxuh/storage/user/lxuh/data/ScanNet++/data"
else
    echo "Dataset not recognized!"
    exit 1
fi

OUTPUT_PATH="output"
CONFIG_PATH="configs/${dataset}"
# EXPERIMENT_NAME="select_hyperparameters"
EXPERIMENT_NAME="ablation_without_color_mask"
SCENE_NAME=${scenes[$SLURM_ARRAY_TASK_ID]}

source ~/miniconda3/etc/profile.d/conda.sh # please change accordingly
conda activate gslam

# Build these libraries specifically for each GPU type, may takes 4-5 min, comment this part if you only work on one node.
pip install -e /home/stud/lxuh/Gaussian-SLAM/gaussian_rasterizer/.
pip install -e /home/stud/lxuh/Gaussian-SLAM/simple-knn/.

echo "Job for dataset: $dataset, scene: $SCENE_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"

# pkill Xvfb
# Xvfb :99 -screen 0 500x500x24 &
# export DISPLAY=:99
# xvfb-run -a

# Your command to run the experiment
srun python run_slam.py "${CONFIG_PATH}/${SCENE_NAME}.yaml" \
                   --input_path "${INPUT_PATH}/${SCENE_NAME}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/${SCENE_NAME}" \
                   --group_name "${EXPERIMENT_NAME}" \

echo "Job for scene $SCENE_NAME completed."
echo "Started at: $START_TIME"
echo "Finished at: $(date)"

# pkill Xvfb

