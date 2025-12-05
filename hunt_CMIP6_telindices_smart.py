#!/bin/bash
"""
WP5 in OptimESM: an algorithm for the detection of tipping elements in CMIP6 model runs

Created by:
Valerio Lembo (CNR-ISAC): v.lembo@isac.cnr.it

Versions:
7/3/25: branching from hunt_CMIP6.py for the detection in teleconnection indices
9/4/25: a code for the detection of tipping in the blocking indices computed via MiLES. Still a bit clumsy, but hopefully working with the hjelp of jinja.
"""

from cdo import Cdo
from math import floor
Cdo.debug = True
from netCDF4 import Dataset as ds
import diptest
import optim_esm_tools as oet
from scipy import stats
import xarray as xr
import yaml
import sys
#Imports for MiLES
from miles.diag import da12_blocking, sc04_blocking, tm90_blocking
from miles.performance import setup_dask
from miles.handling import get_input_directory, get_output_directory
from miles.netcdf import write_output_netcdf, dataset_preprocessing
import dask
from dask.distributed import Client
import lib_optim
from jinja2 import Environment, FileSystemLoader, Template
# import rpy_symmetry as rsym
import datetime
import logging
import os
import re
import shutil
from pathlib import Path
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#Introductory section. 
# client = Client()
cdo = Cdo()
now = datetime.datetime.now()
date = now.isoformat()
logfilen = 'log_hunt_{}.log'.format(date)
logging.basicConfig(filename=logfilen, 
                    level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                   )


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

## User's options
# Fundamental parameters
bandwidth = 10      # for moving average, in years
yravg = bandwidth/2
yrmaxchange = 10        # Chunk length for maximum jumps evaluation
minnofyears = 11        # Shortest dataset to be considered
in_year = 1850          # Initial year for historical/ssp simulations
end_year = 2010         # Last year for ssp simulations
pc = [1, 5, 10, 25,     # set of percentiles for tail comparisons
      75, 90, 95, 99]
thres_gp = 150           # Minimal area (in gridpoints 1x1) for cluster retrieval
plev= 50000

# Defining input/output paths, scenarios, variables and models to be analysed
base_path = Path('/work_big/datasets/synda/data/CMIP6/')
tmp_path = Path('/work_big/users/clima/lembo/tmp/')
figures_path = Path('/home/lembo/tipping_optimesm/figures_{}'.format(date))
try:
    os.makedirs(figures_path)
except OSError:
    pass

projects = ['CMIP', 'ScenarioMIP']
scenarios = [
    {'CMIP':[]},
    # {'ScenarioMIP': ['ssp585', 'ssp370', 'ssp245', 'ssp126']}
    {'ScenarioMIP': ['ssp585']}
    ]
runs = ['r1i1p1f1']
model_groups = [
    # 'BCC',
    # 'CCCma',
    # 'IPSL', 
    # 'MPI-M', 
    'MRI',
    # 'NOAA-GFDL'
    ]
vars = [
        {'day':['psl', 'zg']},
        # {'Lmon':['mrro', 'mrso']},
        # {'SImon':['siconc']},
        # {'Amon':[]},
        # {'Omon':['sos']},
        {'Lmon':[]},
        {'SImon':[]},
        {'Amon':[]},
        {'Omon':[]}]
domains = ['day', 'Lmon', 'SImon', 'Amon', 'Omon']

#setup dask
# dask.config.set(schedule="synchronous")
dask.config.set(scheduler="threads")


def selyear_pi(data, vname, path):
    # Compile a regular expression to match the filename structure.
    # The pattern assumes your filenames strictly follow:
    # vv_dom_m_ss_rr_gg_yyyymmdd-YYYYMMDD.nc
    vv = vname
    dom = data["freq"]
    mm = data["model"]
    ee = data["experiment"]
    rr = data["run"]
    gg = data["grid"]
    
    pattern_str = rf"{vv}_{dom}_{mm}_{ee}_{rr}_{gg}_(\d{{8}})-(\d{{8}})\.nc"
    pattern = re.compile(pattern_str)
    # Initialize lists to hold the start and end years.
    start_years = []
    end_years = []
    # Loop over every file in the folder.
    for filename in os.listdir(path):
        match = pattern.match(filename)
        if match:
            start_date, end_date = match.groups()
            # Extract the first 4 characters as the year.
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
            start_years.append(start_year)
            end_years.append(end_year)
    # Compute the lowest starting year and highest ending year if files were found.
    if start_years and end_years:
        lowest_year = min(start_years)
        highest_year = max(end_years)
        return lowest_year, highest_year
    else:
        logger.info("No matching files found.")
        return None

# Process piControl and historical separately
def process_pi_control(data_context, vname):

    data_pic = data_context.copy()
    data_pic.update({'scenario': 'CMIP'})
    cmip_dir = base_path / data_pic['scenario']
    cmg_dir = cmip_dir / data_pic['mgroup']
    model_dir = cmg_dir / data_pic['model']
    if 'piControl' not in [d.name for d in model_dir.iterdir() if d.is_dir()]:
        logger.info("No piControl run found")
        return

    data_pic.update({'experiment': 'piControl'})
    pi_dir = model_dir / 'piControl' / data_context['run'] 
    grid_dir = pi_dir / data_context['freq'] / vname
    # If piv_dir not found, try alternative frequency folder
    if not pi_dir.exists():
        grid_dir = pi_dir / 'Amon' / vname / data_context['grid']
    if not grid_dir.exists():
       logger.info("PiControl does not have the requested variable")
       return
    
    for grid_path in grid_dir.iterdir():
        if grid_path.is_dir():
            logger.info('GRID: %s', grid_path.name)
            for vers_path in grid_path.iterdir():
                if vers_path.is_dir() and list(vers_path.iterdir()):
                    logger.info('VERS: %s', vers_path.name)
                    piv_dir = grid_dir / grid_path.name / vers_path.name
                    data_pic.update({'grid': grid_path.name, 'vers': vers_path.name})

    data_hist = data_context.copy()
    data_hist.update({'experiment': 'historical'})
    hist_dir = model_dir / 'historical' / data_context['run'] 
    grid_dir = hist_dir / data_context['freq'] / vname
    if not hist_dir.exists():
        grid_dir = hist_dir / 'Amon' / vname / data_context['grid']
    for grid_path in grid_dir.iterdir():
        if grid_path.is_dir():
            logger.info('GRID: %s', grid_path.name)
            for vers_path in grid_path.iterdir():
                if vers_path.is_dir() and list(vers_path.iterdir()):
                    logger.info('VERS: %s', vers_path.name)
                    hiv_dir = grid_dir / grid_path.name / vers_path.name
                    data_hist.update({'grid': grid_path.name, 'vers': vers_path.name})
    
    if piv_dir.exists() and list(piv_dir.iterdir()) and hiv_dir.exists() and list(hiv_dir.iterdir()):
        years = selyear_pi(data_pic, vname, piv_dir)
        # print(years)
        years1=str(years[0])
        years2= str(years[1])
        logger.info('Now processing piControl data (good luck!)...')
        data_pic.update({'inyear': years1.zfill(4), 'outyear': years2.zfill(4)})
        logger.info("Information for the namelist: %s", data_pic)
        # print(data_pic)
        block_pi = render_and_process(data_pic)
        if block_pi is None:
           return
            
        # Process historical run similarly
        # years = selyear_pi(data_context, vname, hiv_dir)
        # logger.info('Now processing historical data...')
        # data_hist.update({'inyear': years[0], 'outyear': years[1]})
        # logger.info("Information for the namelist: %s", data_hist)
        # block_hi = render_and_process(data_hist)
        block_hi = 0
        if block_hi is None:
            return
        else:
            # block = xr.merge([block_pi, block_hi])
            return block_pi
    else:
        logger.info('The piControl run does not contain the requested variable.')
        return None


# Helper: Render the template and process indices
def render_and_process(data, template_file='namelist_tmpl.yml'):
    try:
        logger.info("Information for the namelist: %s", data)
        env = Environment(loader=FileSystemLoader('./'))
        template = env.get_template(template_file)
        rendered_yaml = template.render(data)
        output_filename = f"{data['model']}_namelist.yml"
        with open(output_filename, 'w') as f:
            f.write(rendered_yaml)
        logger.info("Rendered file: %s", output_filename)
        return lib_optim.tel_indices(logger, output_filename, figures_path)
        # return None
    except TypeError as err:
        logger.error("TypeError when processing %s: %s", data, err)
        return None


# Process a single grid level
def process_grid(grid_dir: Path, dom, vv, in_year, end_year, data_context):
    for grid_path in grid_dir.iterdir():
        if grid_path.is_dir():
            logger.info('GRID: %s', grid_path.name)
            for vers_path in grid_path.iterdir():
                if vers_path.is_dir() and list(vers_path.iterdir()):
                    logger.info('VERS: %s', vers_path.name)
                    f_dir = grid_dir / grid_path.name / vers_path.name
                    data_context.update({'grid': grid_path.name, 'vers': vers_path.name})
                    years = selyear_pi(data_context, vv, f_dir)
                    logger.info("Detected years interval: %s", years)
                    logger.info("Input directory: %s", f_dir)
                    data_context.update({'inyear': years[0], 'outyear': years[1]})
                    block = render_and_process(data_context, 'namelist_tmpl.yml')
                    if block is None:
                        continue
                    else:
                        logger.info("What's in the block xarray dataset: %s", block)
                        years_all = [ds.year.values for ds in block]
                        # print(years_all)
                        tm90_all = [ds.tm90.compute().values for ds in block]
                        lib_optim.plot_tser(figures_path, years_all[0], tm90_all[0],
                                            vv, vers_path.name, data_context['model'],
                                            data_context['experiment'], 'eur')
                        lib_optim.plot_tser(figures_path, years_all[1], tm90_all[1],
                                            vv, vers_path.name, data_context['model'],
                                            data_context['experiment'], 'pac')
                    block_pi = process_pi_control(data_context.copy(), vv)
                    if block_pi is None:
                        continue
                    else:
                        logger.info("What's in the blockpi xarray dataset: %s", block_pi)
                        years_all_pi = [ds.year.values for ds in block_pi]
                        # print(years_all_pi)
                        tm90_all_pi = [ds.tm90.compute().values for ds in block_pi]
                        lib_optim.plot_tser(figures_path, years_all_pi[0],
                                            tm90_all_pi[0], vv, vers_path.name,
                                            data_context['model'], 'piC', 'eur')
                        lib_optim.plot_tser(figures_path, years_all_pi[1],
                                            tm90_all_pi[1], vv, vers_path.name,
                                            data_context['model'], 'piC', 'pac')
                        # Compute tipping indicators
                        # block_pi = 0.
                        try:
                            indicators_eur, indicators_pac = lib_optim.tips(logger, block, block_pi, vv, yrmaxchange, pc)
                        except TypeError as e:
                            logger.error("Error computing tipping: %s", e)
                            return
                    yield block, block_pi, vv  # yield block for further processing (e.g., piControl handling)
                    

# Process variables for a particular domain folder
def process_variable(var_dir: Path, domain, var_list, in_year, end_year, base_data):
    for vv in var_list:
        logger.info('VAR: %s', vv)
        # Check if the variable folder exists inside the domain dir
        var_path = var_dir / vv
        if var_path.exists() and list(var_path.iterdir()):
            grids = [g for g in var_path.iterdir() if g.is_dir()]
            for block, block_pi, vv in process_grid(var_path, domain, vv, in_year, end_year, base_data.copy()):
                # In the outer scope the returned block can then trigger the piControl processing
                yield block, block_pi, vv
        else:
            logger.info("%s variable is not available", vv)

# Process runs and domains for a given scenario directory
def process_run(scenario_dir: Path, scenario, run_name, in_year, end_year, data_context):
    run_dir = scenario_dir / run_name
    if not (run_dir.exists() and list(run_dir.iterdir())):
        logger.info("Run %s is not available", run_name)
        return
    logger.info('RUN: %s', run_name)
    for domain in domains:
        domain_dir = run_dir / domain
        if domain_dir.exists() and list(domain_dir.iterdir()):
            logger.info('DOMAIN: %s', domain)
            # Get variables to process for this domain
            # Assuming the order of 'vars' corresponds to domains in DOMAINS
            idx = domains.index(domain)
            var_config = vars[idx].get(domain, [])
            # Update context
            data_context.update({'freq': domain})
            # Process each variable in the list
            for block, block_pi, vv in process_variable(domain_dir, domain, var_config, in_year, end_year, data_context.copy()):
                yield block, block_pi, vv
                # print(block)
                # print(block_pi)
        else:
            logger.info('No variable in %s domain', domain)

def process_scenario(model_dir: Path, scenarios_dict: dict, in_year, end_year, model, mg, mip):
    available_scenarios = [d.name for d in model_dir.iterdir() if d.is_dir()]
    for scenario in scenarios_dict.get(mip, []):
        if scenario in available_scenarios:
            scenario_path = model_dir / scenario
            logger.info('SCENARIO: %s', scenario)
            for run in runs:
                data_context = {
                    'scenario': mip,
                    'mgroup': mg,
                    'model': model,
                    'experiment': scenario,
                    'run': run,
                    'freq': 'day',
                    'grid': 'gg',
                    'vers': 'vers',
                    'inyear': in_year,
                    'outyear': end_year
                }
                # Process runs and domains under the scenario
                for block, block_pi, vv in process_run(scenario_path, scenario, run, in_year, end_year, data_context):
                    # After processing the regular run, attempt the piControl processing using the returned block
                    logger.info('Starting to process experiments in %s...', scenario)
                    # print(block)
                    # process_pi_control(data_context, vv)
        else:
            logger.info("Experiment %s is not available", scenario)

def process_model_group(mip: str, mg: str, in_year, end_year, scenarios_dict):
    mg_dir = base_path / mip / mg
    if not (mg_dir.exists() and list(mg_dir.iterdir())):
        logger.info("No model is found in %s model group", mg)
        return
    logger.info('MODEL GROUP: %s', mg)
    for model in [d.name for d in mg_dir.iterdir() if d.is_dir()]:
        logger.info('MODEL: %s', model)
        # Create output directory for model if needed
        model_out_dir = figures_path / model
        model_out_dir.mkdir(exist_ok=True)
        # Process scenarios inside the model directory
        process_scenario(mg_dir / model, scenarios_dict, in_year, end_year, model, mg, mip)

# Main loop processing each project (MIP)
def process_projects(in_year, end_year, scenarios_list):
    # Convert list of scenarios dictionaries into a lookup
    scenario_lookup = {}
    for d in scenarios_list:
        scenario_lookup.update(d)

    logger.info('Starting the loops on the files...')
    for mip in projects:
        mip_dir = base_path / mip
        if not (mip_dir.exists() and list(mip_dir.iterdir())):
            logger.info("The MIP is empty")
            continue
        logger.info('MIP: %s', mip)
        for mg in model_groups:
            process_model_group(mip, mg, in_year, end_year, scenario_lookup)
    logger.info("Finished hunting for tipping. Now rest...")

# Run the main process with the given time boundaries
process_projects(in_year, end_year, scenarios)

# mipp=0
# logger.info('Starting the loops on the files...')
# for mip in project:
#     mip_dir = os.path.join(path, mip)
#     if os.path.isdir(mip_dir) and os.listdir(mip_dir):
#         logger.info('MIP: {}'.format(mip))
#         for mg in model_groups:
#             mg_dir = os.path.join(mip_dir, mg)
#             if os.path.isdir(mg_dir) and os.listdir(mg_dir):
#                 logger.info('MODEL GROUP: {}'.format(mg))
#                 models = [d for d in os.listdir(mg_dir)]
#                 for m in models:
#                     logger.info('MODEL: {}'.format(m))
#                     s_dir = os.path.join(mg_dir, m)
#                     f_dir = os.path.join(path_l,m)
#                     try:
#                         os.makedirs(f_dir)
#                     except OSError:
#                         pass
#                     scens = os.listdir(s_dir)
#                     logger.info("In MODEL DIR: {}".format(scens))
#                     # for i in np.arange(len(scenarios[mipp][mip])):
#                     for ss in scenarios[mipp][mip]:
#                         # if scenarios[mipp][mip][i] in scens:
#                         if ss in scens:
#                             m_dir = os.path.join(s_dir, ss)
#                             if os.path.isdir(m_dir) and os.listdir(m_dir):
#                                 logger.info('SCENARIO: {}'.format(ss))
#                                 run = [r for r in os.listdir(m_dir)]    
#                                 for rr in runs:
#                                     if rr in run:
#                                         logger.info('RUN: {}'.format(rr))
#                                         j=0
#                                         for dom in domains:
#                                             logger.info('DOMAIN: {}'.format(dom))
#                                             if dom in os.listdir(os.path.join(m_dir, rr)):
#                                                 v_dir = os.path.join(m_dir, rr, dom)
#                                                 var = [v for v in os.listdir(v_dir)]
#                                                 for vv in vars[j][dom]:
#                                                     logger.info('VAR: {}'.format(vv))
#                                                     if vv in var:
#                                                         g_dir = os.path.join(v_dir, vv)
#                                                         grids = [g for g in os.listdir(g_dir)]
#                                                         for gg in grids:
#                                                             logger.info('GRID: {}'.format(gg))
#                                                             ve_dir = os.path.join(g_dir,gg)
#                                                             vers = [ve for ve in os.listdir(ve_dir)]
#                                                             for vee in vers:
#                                                                 logger.info('VERS: {}'.format(vee))
#                                                                 f_dir = os.path.join(ve_dir, vee)
#                                                                 if os.path.isdir(f_dir) and os.listdir(f_dir):
#                                                                     try:
#                                                                         data = {
#                                                                             'scenario': mip,
#                                                                             'mgroup': mg,
#                                                                             'model': m,
#                                                                             'experiment': ss,
#                                                                             'run': rr,
#                                                                             'freq': dom,
#                                                                             'grid': gg,
#                                                                             'vers': vee,
#                                                                             'inyear': in_year,
#                                                                             'outyear': end_year
#                                                                         }
#                                                                         env = Environment(loader=FileSystemLoader('./'))
#                                                                         template = env.get_template('namelist_tmpl.yml')
#                                                                         rendered_yaml = template.render(data)
#                                                                         output_filename = f"{data['model']}_namelist.yml"
#                                                                         with open(output_filename, 'w') as f:
#                                                                             f.write(rendered_yaml)
#                                                                         print(output_filename)
#                                                                         block = lib_optim.tel_indices(output_filename)
#                                                                     except TypeError:
#                                                                         continue
#                                                                     cmip_dir = os.path.join(path, 'CMIP')
#                                                                     cmg_dir = os.path.join(cmip_dir, mg)
#                                                                     if 'piControl' in os.listdir(os.path.join(cmg_dir, m)):
#                                                                         # logger.info("The piControl is present...")
#                                                                         pi_dir = os.path.join(cmg_dir, m, 'piControl', rr)
#                                                                         hi_dir = os.path.join(cmg_dir, m, 'historical', rr)
#                                                                         piv_dir = os.path.join(pi_dir, dom, vv, gg)
#                                                                         if not os.path.isdir(piv_dir):
#                                                                             piv_dir = os.path.join(pi_dir, 'Amon', vv, gg)
#                                                                         hiv_dir = os.path.join(hi_dir, dom, vv, gg)
#                                                                         if os.path.isdir(piv_dir) and os.listdir(piv_dir) and os.path.isdir(hiv_dir) and os.listdir(hiv_dir):   
#                                                                             # if dom in os.listdir(pi_dir) and dom in os.listdir(hi_dir):
#                                                                             if dom in os.listdir(hi_dir):
#                                                                                 vers = [v for v in os.listdir(piv_dir)]
#                                                                                 vep = vers[0]
#                                                                                 fpi_dir = os.path.join(piv_dir, vep)
#                                                                                 vers = [v for v in os.listdir(hiv_dir)]
#                                                                                 veh = vers[0]
#                                                                                 fhi_dir = os.path.join(hiv_dir, veh)
#                                                                                 if os.path.isdir(fpi_dir) and os.listdir(fpi_dir):
#                                                                                     try:
#                                                                                         logger.info('Now crunching piControl data (good luck!)...')
#                                                                                         data = {
#                                                                                             'scenario': mip,
#                                                                                             'mgroup': mg,
#                                                                                             'model': m,
#                                                                                             'experiment': ss,
#                                                                                             'run': rr,
#                                                                                             'freq': dom,
#                                                                                             'grid': gg,
#                                                                                             'vers': vee,
#                                                                                             'inyear': in_year,
#                                                                                             'outyear': end_year
#                                                                                         }
#                                                                                         env = Environment(loader=FileSystemLoader('./'))
#                                                                                         template = env.get_template('namelist_tmpl.yml')
#                                                                                         rendered_yaml = template.render(data)
#                                                                                         output_filename = f"{data['model']}_namelist.yml"
#                                                                                         with open(output_filename, 'w') as f:
#                                                                                             f.write(rendered_yaml)
#                                                                                         print(output_filename)
#                                                                                         block_pi = lib_optim.tel_indices(output_filename)
#                                                                                         logger.info('piControl data crunched!')
#                                                                                     except TypeError:
#                                                                                         continue
#                                                                                     #Computing tipping indicators
#                                                                                     try:
#                                                                                         [indicators, masks] = lib_optim.tips(block, block_pi, 
#                                                                                                                              vv, yrmxch)
#                                                                                     except TypeError as e:
#                                                                                         continue
#                                                                                     try:
#                                                                                         data = {
#                                                                                             'scenario': mip,
#                                                                                             'mgroup': mg,
#                                                                                             'model': m,
#                                                                                             'experiment': ss,
#                                                                                             'run': rr,
#                                                                                             'freq': dom,
#                                                                                             'grid': gg,
#                                                                                             'vers': vee,
#                                                                                             'inyear': in_year,
#                                                                                             'outyear': end_year
#                                                                                         }
#                                                                                         env = Environment(loader=FileSystemLoader('./'))
#                                                                                         template = env.get_template('namelist_tmpl.yml')
#                                                                                         rendered_yaml = template.render(data)
#                                                                                         output_filename = f"{data['model']}_namelist.yml"
#                                                                                         with open(output_filename, 'w') as f:
#                                                                                             f.write(rendered_yaml)
#                                                                                         print(output_filename)
#                                                                                         block_hi = lib_optim.tel_indices(output_filename)
#                                                                                     except TypeError:
#                                                                                         continue
#                                                                                     lonm = np.array(lon)
#                                                                                     latm = np.array(lat)
#                                                                                     path_f = '{}/{}/{}/{}'.format(path_l, m, ss, vv) 
#                                                                                     try:
#                                                                                         os.makedirs(path_f)
#                                                                                     except OSError:
#                                                                                       pass
                                                                                    
#                                                                                     # os.remove(ofile_piy)
#                                                                                     # os.remove(ofile_pistd)
#                                                                                 else:
#                                                                                     logger.info('Directory is empty...')
#                                                                         else:
#                                                                             logger.info('The piControl run does not contain the requested variable.')
#                                                                     else:
#                                                                         logger.info('No piControl run found')
#                                                                     # os.remove(ofile_y)
#                                                                     # os.remove(ofile_std)
#                                                                 else:
#                                                                     logger.info('The directory is empty...')
#                                                     else:
#                                                         logger.info("{} variable is not available".format(vv))
#                                             else:
#                                                 logger.info('No variable in {} domain'.format(dom))
#                                             j = j+1
#                                     else:
#                                         logger.info("Run {} is not available".format(rr))
#                             else:
#                                 logger.info("Theere's nothing in this scenario.")
#                         else:
#                             logger.info("Experiment {} is not available".format(ss))
#             else:
#                 logger.info("No model is found in {} model group".format(mg))
#     else:
#         logger.info("The MIP is empty")
#     mipp = mipp+1
# logger.info("Finished hunting for tipping. Now rest...")
