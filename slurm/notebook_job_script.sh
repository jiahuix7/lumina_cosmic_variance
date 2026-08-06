#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --exclude=node1431,node1434
#SBATCH --constraint=rocky8
#SBATCH --mem=64G
#SBATCH --output=jupyter_%j.out
#SBATCH --export=NONE

if [ -f /etc/bashrc ];then
    . /etc/bashrc
fi

module purge
module load community-modules
module load miniforge/24.3.0-0
conda activate lumina_cv

cd /orcd/home/002/jxie2/Projects/lumina_cosmic_variance

echo "compute node:"
hostname
which python

jupyter lab notebooks/sfe_models.ipynb --no-browser --ip=0.0.0.0 --port=8563
