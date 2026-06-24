#!/bin/bash
#SBATCH --job-name=lumina_hmf_cv
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=jxie2@caltech.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"

cd "${HOME}/Projects/lumina_cosmic_variance"
mkdir -p logs figures outputs

module purge
module load community-modules
module load miniforge/24.3.0-0

conda activate mypy3
which python
python --version

cd notebooks
python compute_cv.py

echo "Job finished"
