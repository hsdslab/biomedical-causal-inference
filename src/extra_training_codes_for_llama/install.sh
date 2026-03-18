#!/usr/bin/env bash
# ==============================================================================
# Script to setup the conda environment for Llama training on a SLURM cluster.
# Installs dependencies listed in requirements.txt
# ==============================================================================
#SBATCH --job-name=llama_install   # Job name
#SBATCH --output=logs/install_%j.out      # Standard output log (%j expands to jobID)
#SBATCH --error=logs/install_%j.err       # Standard error log
#SBATCH --partition=gpu                    # Partition (queue) with GPU access
#SBATCH --cpus-per-task=1                  # Number of CPU cores per task
#SBATCH --mem-per-cpu=48G                        # Total memory per node
#SBATCH --time=1:00:00                    # Time limit (hh:mm:ss)

# Load necessary modules or environment
module load miniconda3
# Activate your Python environment
source ~/.bashrc
conda remove -n adenew --all -y
conda clean --all -y
conda clean --all -y # Also clean the cache from any corrupted downloads

conda create -n adenew python=3.11 -y 
conda activate adenew
nvidia-smi
pip install -r requirements.txt
