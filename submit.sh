#!/bin/bash
#
#SBATCH --job-name=NewsGraphing
#SBATCH --output=ng.res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=60:00
#SBATCH --mem-per-cpu=8G

# load cluster specific modules to get software on the path
module load anaconda3
module load julia
# load python specific conda environment to get python packages on the path
source activate my_root

cd ~/NewsGraphing
srun python main.py
