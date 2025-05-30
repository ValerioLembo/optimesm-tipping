# Discovery of Moderate Non-linear Surprises (MNS) in CMIP6 model outputs

An algorithm to detect moderate non-linear surprises in CMIP6 model runs and beyond.

## Requirements
- Anaconda/Miniconda/Micromamba
- Python >= 3.11

## Configuration
- conda env create -f env_telindices.yml
- Declarations are at lines 58-

## Usage
1. conda activate optimesm-tipping
2. Launch serial or parallel job:
   - Sequential: python3 hunt_CMIP6.py (or hunt_CMIP76_telindices_smart.py)
   - Parallel: sbatch hunt_CMIP6.py (or hunt_CMIP76_telindices_smart.py)

## Design
Variables under consideration are labeled according to CF-conventions for standard short variable names in CMIP6 model outputs, i.e.
evspsbl (evaporation flux at the surface), mrro (total runoff), mrso (total soil moisture), mrsos (total soil
moisture at the surface), pr (rainfall precipitation), siconc (sea-ice concentration), sos (surface salinity),
tas (near-surface air temperature), tos (sea surface temperature). Each quantity is examined at every
gridpoint for existence of a MNS in the local time series of its anomaly. Neighboring gridpoints meeting
at least three out of the five chosen criteria are clustered into regions (column 6). Criteria are labeled as
such: i) Maximum 10-years jump in the scenario exceeding 99-percentile of the distribution of 10-years
jumps in the preindustrial scenario; ii) rejection of null hypothesis for a Dip multimodality test with
0.05 confidence level; iii) standard deviation of the forced scenario exceeding the 99-percentile of the
distribution of gridpointwise standard deviations in the preindustrial run; iv) all annual mean anomalies
in the last 30 years of the forced scenario exceeding the 1- or 99-percentile of the distribution of anomalies
in the preindustrial scenario; v) rejection of null hypothesis for a Kolmogorov-Smirnov normality test with
a 0.05 confidence level. Criteria met for the spatially averaged anomaly time series in the clustered region
are displayed in column 4.
