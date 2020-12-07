#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --partition=batch
#SBATCH -J geoai
#SBATCH -o g20_gbr_.%J.out
#SBATCH -e g20_gbr_.%J.out
#SBATCH --time=10:00:00
#SBATCH --mem=350G
#SBATCH --constraint=rome

. "/home/omairyrm/anaconda3/etc/profile.d/conda.sh"
export PATH="/home/omairyrm/anaconda3/bin:$PATH"
conda activate
python gbr_rabab.py 2 -v
