#!/bin/bash
#SBATCH --job-name=lumina_smf_cv
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures outputs

module purge
module load community-modules
module load miniforge/24.3.0-0

conda activate lumina_cv
which python
python --version

python scripts/compute_UVLF_cv.py

echo "Job finished"
