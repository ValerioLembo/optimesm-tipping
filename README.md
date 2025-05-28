# Discovery of Moderate Non-linear Surprises (MNS) in CMIP6 model outputs

An algorithm to detect moderate non-linear surprises in CMIP6 model runs and beyond.

## Requirements
- Anaconda/Miniconda/Micromamba
- Python >= 3.11

## Configuration
- conda env create -f env_telindices.yml

## Usage
1. conda activate optimesm-tipping
2. Launch serial or parallel job:
   - Sequential: python3 hunt_CMIP6.py (or hunt_CMIP76_telindices_smart.py)
   - Parallel: sbatch hunt_CMIP6.py (or hunt_CMIP76_telindices_smart.py)

## Design
TBC
