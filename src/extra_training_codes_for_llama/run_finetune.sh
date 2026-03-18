#!/usr/bin/env bash
# ==============================================================================
# SLURM script to execute the LLaMA fine-tuning process.
# Runs llama_train.py sequentially with optimal thread settings.
# ==============================================================================
#SBATCH --job-name=llama_finetune    # Job name
#SBATCH --output=logs/llama_%j.out      # Standard output log (%j expands to jobID)
#SBATCH --error=logs/llama_%j.err       # Standard error log
#SBATCH --partition=gpu                   # Partition (queue) with GPU access
#SBATCH --nodelist=scc-gpu01-10G
#SBATCH --cpus-per-task=1                  # Number of CPU cores per task
#SBATCH --mem-per-cpu=48G                        # Total memory per node
#SBATCH --time=240:00:00                    # Time limit (hh:mm:ss)

# Load necessary modules or environment
module load miniconda3
# Activate your Python environment
source ~/.bashrc
conda activate adenew
nvidia-smi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=4

# Run the inference script
srun python llama_train.py
