#!/bin/bash 
#SBATCH --account=rrg-ashique
#SBATCH --cpus-per-task=2
#SBATCH --array=0-29
#SBATCH --time=0-07:50
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=fshahri1@ualberta.ca
#SBATCH --mail-type=ALL 

module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
source /home/fshahri1/projects/def-ashique/fshahri1/jsac_tests/JSACENV/bin/activate

export PYTHONPATH=/home/fshahri1/projects/def-ashique/fshahri1/jsac_tests/JSAC:$PYTHONPATH

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 /home/fshahri1/projects/def-ashique/fshahri1/jsac_tests/JSAC/tasks/simulation/task_mujoco.py --seed=$SLURM_ARRAY_TASK_ID --env_name="Walker2d-v4"
sleep 10
python3 /home/fshahri1/projects/def-ashique/fshahri1/jsac_tests/JSAC/tasks/simulation/task_mujoco.py --seed=$SLURM_ARRAY_TASK_ID --env_name="Humanoid-v4"
sleep 10
python3 /home/fshahri1/projects/def-ashique/fshahri1/jsac_tests/JSAC/tasks/simulation/task_mujoco.py --seed=$SLURM_ARRAY_TASK_ID --env_name="Hopper-v4"
sleep 10
python3 /home/fshahri1/projects/def-ashique/fshahri1/jsac_tests/JSAC/tasks/simulation/task_mujoco.py --seed=$SLURM_ARRAY_TASK_ID --env_name="HalfCheetah-v4"
sleep 10
python3 /home/fshahri1/projects/def-ashique/fshahri1/jsac_tests/JSAC/tasks/simulation/task_mujoco.py --seed=$SLURM_ARRAY_TASK_ID --env_name="Ant-v4"
sleep 10