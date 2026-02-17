#!/bin/bash
#SBATCH -J Venra    # Name that will show up in squeue
#SBATCH --gres=gpu:1         # Request 4 GPU "generic resource"
#SBATCH --time=3-00:00:00       # Max job time is 3 hours
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --nodelist=cs-venus-09   # if needed, set the node you want (similar to -w xyz)
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8


# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate .env
# conda activate hcl-env
hostname
echo $CUDA_AVAILABLE_DEVICES
#export OMP_NUM_THREADS=1

# Note the actual command is run through srun
cd /localscratch/pagand/VeNRA
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYOPENGL_PLATFORM=egl
nvidia-smi
ulimit -u 1029439

srun python -u src/hal_det/training/train.py  --output_dir ./data/output