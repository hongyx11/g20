#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --partition=batch
#SBATCH -J geoai
#SBATCH -o geoai_46_.%J.out
#SBATCH -e geoai_46_.%J.out
#SBATCH --time=02:00:00
#SBATCH --mem=64G

source ~/miniconda3/bin/activate

python play_yuxi.py 4
python play_yuxi.py 6