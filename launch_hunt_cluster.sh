#!/bin/bash
#SBATCH -N 1                # 2 nodes
#SBATCH -n 16
#SBATCH --mem=7100M
#SBATCH -p batch
#SBATCH --time 18:00:00
#SBATCH --job-name=optimesm-tipping
#SBATCH --mail-user=valerio.lembo@cnr.it
#SBATCH --output=/work/users/clima/lembo/log/hunt_CMIP6_log-%j.out
#SBATCH --error=/work/users/clima/lembo/log/hunt_CMIP6_log-%j.err

unset DISPLAY
export MPLBACKEND=Agg

source activate jupyterlab_valerio_ok
python3 hunt_CMIP6.py
