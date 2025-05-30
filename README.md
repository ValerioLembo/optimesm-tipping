# Discovery of Moderate Non-linear Surprises (MNS) in CMIP6 model outputs

An algorithm to detect moderate non-linear surprises in CMIP6 model runs and beyond.

## Requirements
- Anaconda/Miniconda/Micromamba
- Python >= 3.11

## Configuration
- conda env create -f env_telindices.yml
- User's options can be inserted at lines 57-75. Some considerations:
   - Dataset's folder has to be organized as a ESGF node (Project/Model_group/Model/Scenario/Run/Domain/Variable/Grid/Version)
   - Scenarios are organized by Project (e.g. ssp scenarios are under ScenarioMIP folder)
   - Variables are organized by domain (daily fields can be ingested) and labeled as short variable names; 

## Usage
1. conda activate optimesm-tipping
2. Launch serial or parallel job:
   - Sequential: python3 hunt_CMIP6.py (or hunt_CMIP76_telindices_smart.py)
   - Parallel: sbatch hunt_CMIP6.py (or hunt_CMIP76_telindices_smart.py)

## Design
The algorithms scans through available datasets withing each list of scenarios, runs, model groups, variables provided by the user. First, the dataset is aggregated along time, latxlon fields are interpolated on a regular 1 x 1 degress grid. A 10-years running mean is applied at every gridpoint, and the existence of a MNS in the local time series of its anomaly. In order to do so, five criteria are considered: i) Maximum 10-years jump in the scenario exceeding 99-percentile of the distribution of 10-years
jumps in the preindustrial scenario; ii) rejection of null hypothesis for a Dip multimodality test with 0.05 confidence level; iii) standard deviation of the forced scenario exceeding the 99-percentile of the distribution of gridpointwise standard deviations in the preindustrial run; iv) all annual mean anomalies in the last 30 years of the forced scenario exceeding the 1- or 99-percentile of the distribution of anomalies in the preindustrial scenario; v) rejection of null hypothesis for a Kolmogorov-Smirnov normality test with
a 0.05 confidence level. A mask is produced, highlighting all gridpoints for which at least three out of 5 criteria are met. A clustering algorithm is then applied, aggregating neighboring gridpoints. Clusters that are within a given distance from other clusters are considered as a single cluster. Finally, a threshold to the minimal size of the cluster is given. For each considered cluster, an image as the example below is given, with top,left panel denoting the cluster, top,right panel the time series of historical+scenario evolution of anomalies wrt. preindustrial conditions averaged inside the cluster, bottom,left panel time series of standard deviation evolution averaged inside the cluster, bottom,right some data about the detection criteria.

<img width="713" alt="Screenshot 2025-05-30 alle 19 15 54" src="https://github.com/user-attachments/assets/e29ae880-9719-4f2d-baeb-61a2326a26e3" />

A table of detected MNS for a number of observables in CMIP6 ssp scenarios, using this tool, is provided [here](https://doi.org/10.5281/zenodo.15498970).
