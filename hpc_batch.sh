#!/bin/bash
#SBATCH -J eval
#SBATCH -A mime
#SBATCH -t 7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=senewiry@oregonstate.edu
#SBATCH -p dgx
#SBATCH -n 1 # request tasks
#SBATCH -c 2 # request 2 cores
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --mem=20GB # request X GB
#SBATCH -o /nfs/stak/users/senewiry/osu_thesis/outfiles/log.11.out

# Load any required software environment module
. ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# run my job
python3 /nfs/stak/users/senewiry/osu_thesis/main.py eval 11
