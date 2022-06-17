#!/bin/bash
#SBATCH -J eval
#SBATCH -A mime
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=senewiry@oregonstate.edu
#SBATCH -p dgx
#SBATCH -n 1 # request tasks
#SBATCH -c 4 # request 8 cores
#SBATCH -N 2 # number of nodes
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --constraint=v100                  # request node with V100 GPU
#SBATCH --mem=20GB # request X GB
#SBATCH --nodelist=dgx2-5

# Load any required software environment module
. ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# run my job
python3 /nfs/stak/users/senewiry/osu_thesis/main.py eval 1