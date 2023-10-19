"""
WP5 in OptimESM: an algorithm for the detection of tipping elements in CMIP6 model runs

Created by:
Valerio Lembo (CNR-ISAC): v.lembo@isac.cnr.it

Versions:
10/5/23: And so it begins...
08/09/2023: Solved most issues with clustering, now polishing the script...
19/10/23: working version. Combines three masks: std, max jump, pc99
"""

from cdo import Cdo
from math import floor
Cdo.debug = True
from netCDF4 import Dataset as ds
import optim_esm_tools as oet
import rpy_symmetry as rsym
import datetime
import logging
import warnings
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

cdo = Cdo()
now = datetime.datetime.now()
date = now.isoformat()
logfilen = 'log_hunt_{}.log'.format(date)
logging.basicConfig(filename=logfilen, level=logging.INFO, flush=True)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def data_crunch(f_dir,scen,var,filter,in_year,end_year):
    os.chdir(f_dir)
    ofile = os.path.join(f_dir, 'file_merged.nc')
    try:
        os.remove(ofile)
    except OSError:
        pass
    if len(os.listdir(f_dir))>1:
        cdo.mergetime(
            input=glob.glob(f_dir+'/{}_*.nc'.format(var)),
            options = '-O',
            output = ofile)
    else:
        ncfile = os.listdir(f_dir)
        shutil.copy(ncfile[0],ofile)
    ofile_y = os.path.join(f_dir, 'file_merged_y.nc')
    try:
        os.remove(ofile_y)
    except OSError:
        pass
    if scen=='ssp585' or scen=='ssp245' or scen=='ssp126':
        if var=='siconc' or var=='sos' or var=='tos':
            cdo.setmisstoc(0,
                input= '-selyear,{}/{} -runmean,{} -selvar,{} -remapbil,r360x180 -yearmean {}'.format(in_year,end_year,filter,var,ofile),
                options = '-P 8',
                output = ofile_y)
        else:
            cdo.runmean(filter,
                input= '-selyear,{}/{} -remapbil,r360x180 -yearmean {}'.format(in_year,end_year,ofile),
                options = '--pedantic -v -P 8',
                output = ofile_y)
    else:
        if var=='siconc' or var=='sos' or var=='tos':
            cdo.setmisstoc(0,
                input = '-runmean,{} -selvar,{} -remapbil,r360x180 -yearmean {}'.format(filter,var,ofile),
                options = '-P 8',
                output = ofile_y)
        else:
            cdo.runmean(filter,
                input= '-remapbil,r360x180 -yearmean {}'.format(ofile),
                options = '-P 8',
                output = ofile_y)
    ofile_std = os.path.join(f_dir, 'file_merged_std.nc')
    try:
        os.remove(ofile_std)
    except OSError:
        pass                                                    
    cdo.timstd(
        input = '-detrend {}'.format(ofile_y),
        output = ofile_std)
    try:
        os.remove(ofile)
    except OSError:
        pass    
    return ofile_y, ofile_std


def julian_date_to_decimal_years(jd):
    # Calculate Julian centuries (T) since 1850-01-01
    ref_date = datetime.datetime(1850, 1, 1)
    target_date = ref_date + np.vectorize(datetime.timedelta)(days=jd)
    decimal_years = []
    for date in target_date:
        decimal_year = date.year + (date.timetuple().tm_yday -1)/365.25
        decimal_years.append(decimal_year)
    decimal_years = np.array(decimal_years)
    return decimal_years


def map(path, lons, lats, data, var, ver, model, scen, mode, rea):
    m = Basemap(projection='cyl', resolution='c', lon_0=180.)
    lons[lons > 180.] -= 360.
    lons_2 = lons[lons>=0]
    lons_3 = np.append(lons[lons<0], lons_2, 0)
    lons = lons_3 + 180.
    # draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='coral', lake_color='aqua')
    # plot data on the map
    vmax = np.nanmax(np.abs(data))
    vmin = 0.
    m.pcolormesh(lons, lats, np.squeeze(data), cmap='Reds', vmin = vmin, vmax = vmax)
    # add title
    plt.title(var + " " + model + " " + ver + " " + scen + "\n " + mode + " " + rea)
    plt.colorbar()
    # show and save the map
    plt.savefig(path + "/" + model + "/" + var + "_" + model + "_" + ver + "_" + scen + "_" + rea + "_" + mode + "_diff.png")
    plt.close()


def reg_boxsel(lon, lat, data, lomin, lomax, lamin, lamax):
    bnds_1 = np.argmax(lon[lon<lomin])
    lonmin = np.min(lon[lon>lomax])
    bnds_2, = np.where(lon==lonmin)
    bnds_2 = bnds_2[0]
    bnds_3 = np.argmax(lat[lat<-lamin])
    latmin = np.min(lat[lat>lamax])
    bnds_4, = np.where(lat==latmin)
    bnds_4 = bnds_4[0]
    boxsel = data[:,bnds_3:bnds_4,bnds_1:bnds_2]
    boxsel_gm = np.squeeze(np.nanmean(np.nanmean(boxsel,axis=2),axis=1))
    return boxsel_gm


def tips(filein,filein_std,filein_pi,filein_pistd,varname,yrmxch):            
    data = ds(filein)
    var = data.variables[varname][:,:,:]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    time = data.variables['time'][:]
    nyrs = len(time)
    usedyears = 2 * yravg
    if nyrs <= usedyears:
        logger.info('Not enough years')
        return
    fin = nyrs - yrmxch
    data = ds(filein_std)
    var_std = data.variables[varname][:,:,:]
    pcm_std = np.percentile(var_std, pc)
    data = ds(filein_pi)
    time_pi = data.variables['time'][:]
    nyrs_pi = len(time_pi)
    finpi = nyrs_pi - yrmxch
    varpi = data.variables[varname][:,:,:]
    data = ds(filein_pistd)
    varpi_std = data.variables[varname][:,:,:]
    var_shift = np.zeros(np.shape(var))
    var_shift[yrmxch:,:,:] = var[:fin,:,:]
    var_shiftdiff = var[yrmxch:,:,:]-var_shift[yrmxch:,:,:]
    var_diffabs = np.squeeze(np.abs(var_shiftdiff))
    var_timmax = np.nanmax(var_diffabs,0)
    
    # rsym.p_symmetry(values, test_statistic='BHI', **kw)
    
    varpi_shift = np.zeros(np.shape(varpi))
    varpi_shift[yrmxch:,:,:] = varpi[:finpi,:,:]
    varpi_shiftdiff = varpi[yrmxch:,:,:]-varpi_shift[yrmxch:,:,:]
    varpi_diffabs = np.squeeze(np.abs(varpi_shiftdiff))
    varpi_timmax = np.nanmax(varpi_diffabs,0)
    pcm_jump = np.percentile(var_timmax, pc)
    var_ini = var[:yrmxch,:,:] 
    var_end = var[fin:,:,:] 
    var_ini_tm = np.nanmean(var_ini, 0)
    var_end_tm = np.nanmean(var_end, 0)
    pcm = np.percentile(varpi, pc, axis=0)
    mask_std = np.where(
        ((np.abs(var_std)>pcm_std[6]).astype(bool) &
        ((np.abs(varpi_std)!=0.).astype(bool)) &
        (~np.isnan(varpi_std))), 
        1, 0)
    mask_max = np.where(
        ((np.abs(var_timmax)>pcm_jump[6]).astype(bool) & 
        ((np.abs(varpi_timmax)!=0.).astype(bool)) &
        (~np.isnan(varpi_timmax))), 
        1, 0)
    mask_75p = np.where(
        ((var>pcm[4,:,:]).astype(bool) |
         (var<pcm[0,:,:]).astype(bool) &
        (var!=0.).astype(bool) &
        (~np.isnan(var))),
        1, 0)
    mask_90p = np.where(
        ((var>pcm[5,:,:]).astype(bool) |
        (var<pcm[1,:,:]).astype(bool) &
        (var!=0.).astype(bool) &
        (~np.isnan(var))),
        1, 0)
    mask_95p = np.where(
        ((var>pcm[6,:,:]).astype(bool) |
        (var<pcm[2,:,:]).astype(bool) &
        (var!=0.).astype(bool) &
        (~np.isnan(var))),
        1, 0)
    mask_99p = np.where(
        ((var>pcm[7,:,:]).astype(bool) |
        (var<pcm[3,:,:]).astype(bool) &
        (var!=0.).astype(bool) &
        (~np.isnan(var))),
        1, 0)
    mask_combine = np.where(
        ((mask_std.astype(int) + 
          mask_max.astype(int) + 
          np.nansum(mask_99p[-30:,:,:].astype(int),axis=0)/30)==3).astype(bool),
        1, 0)
    indicators = [var_std, var_timmax]
    mask_99ps = np.where(
        ((np.nansum(mask_99p[-30:,:,:].astype(int),axis=0)/30)==1.).astype(bool),
        1, 0)
    masks = [mask_std, mask_max, mask_99ps, mask_combine]
    return time, lon, lat, var, indicators, masks


def plotting_clusters(path, clusters, time, lon, lat, data, vv, vee, mod, method, scen, thres, dom):
    for cl in range(len(clusters[:,0,0])):
        cl_tser = clusters[cl,:,:]
        if np.nansum(cl_tser)>thres:
            logger.info("Cluster {} of {} for criterion {} passes the threshold for the area".format(cl, 
                                                                                                     np.size(clusters,axis=0), 
                                                                                                     method))
            fld = cl_tser[np.newaxis,:,:] * data
            if method == 'std':
                fld_n = np.where(fld==0,np.nan,fld)
                tser = np.nanstd(np.nanstd(fld_n,axis=2),axis=1)
            else:
                crate = np.nansum(np.nansum(cl_tser,axis=0))/(len(lon)*len(lat))
                tser = np.nanmean(np.nanmean(fld,axis=2),axis=1)/crate
            plot_tser(path, time, tser, vv, vee, mod, scen, 'cluster_{}_{}'.format(method, cl), dom)
            map(path, lon, lat, 
                np.squeeze(clusters[cl,:,:]), 
                vv, vee, m, scen, 'clusters_{}_{}'.format(method, cl), dom)
        else:
            pass

                                                                                
def plot_tser(path, time, var, vname, ver, model, scen, name, rea):
    # Plotting
    yrs = julian_date_to_decimal_years(time)
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(yrs, var, color='blue', linewidth=1)
    plt.title(vname + " " + model + " " + ver + " " + scen + "\n" + rea)
    plt.xlabel('Years')  # Set the x-axis label
    plt.ylabel(vname)  # Set the y-axis label
    plt.grid(True)  # Enable gridlines
    plt.tight_layout()  # Adjust the spacing of the plot
    plt.savefig(path + "/" + model + "/" + vname + "_" + model + "_" + ver + "_" + scen + "_" + rea + "_" + name + "_tser.png")
    plt.close()


bandwidth = 10
yravg = 5
yrmaxchange = 10
minnofyears = 11
in_year = 1850
end_year = 2100
pc = [1, 5, 10, 25, 75, 90, 95, 99]
thres_gp = 200
project = ['ScenarioMIP']
model_groups = [
    'AS-RCEC', 'BCC', 'CAMS', 'CCCma', 'CCCR-IITM', 
    'CMCC', 
    'CSIRO', 
    'CSIRO-ARCCSS', 
    'E3SM-Project', 
    'EC-Earth-Consortium', 'FIO-QLNM',
    'HAMMOZ-Consortium', 
    'INM', 
    'IPSL', 
    'KIOST', 'MIROC', 'MOHC', 
    'MPI-M', 
    'MRI',
    'NASA-GISS', 
    'NCAR', 
    'NCC', 'NIMS-KMA',
    'NOAA-GFDL', 
    'NUIST', 'SNU',
    'THU'
    ]
#'CAS', 'UA']
vars = [
        {'SImon':['siconc']},
        {'Amon':['tas']},
        {'Omon':['sos','tos']}]
domains = ['SImon', 'Amon', 'Omon']
scenarios = [
    #{'CMIP':['historical', 'abrupt-4xCO2']},
    {'ScenarioMIP': ['ssp585', 'ssp370', 'ssp245', 'ssp126']}]
runs = ['r1i1p1f1', 'r2i1p1f1']
in_year = 2015
end_year = 2100
path = '/work_big/datasets/synda/data/CMIP6/'
path_l = '/home/lembo/tipping_optimesm/figures_{}'.format(date)
try:
    os.makedirs(path_l)
except OSError:
    pass


mipp=0
logger.info('Starting the loops on the files...')
for mip in project:
    mip_dir = os.path.join(path, mip)
    if os.path.isdir(mip_dir) and os.listdir(mip_dir):
        logger.info('MIP: {}'.format(mip))
        for mg in model_groups:
            mg_dir = os.path.join(mip_dir, mg)
            if os.path.isdir(mg_dir) and os.listdir(mg_dir):
                logger.info('MODEL GROUP: {}'.format(mg))
                models = [d for d in os.listdir(mg_dir)]
                for m in models:
                    logger.info('MODEL: {}'.format(m))
                    s_dir = os.path.join(mg_dir, m)
                    f_dir = os.path.join(path_l,m)
                    try:
                        os.makedirs(f_dir)
                    except OSError:
                        pass
                    scens = os.listdir(s_dir)
                    logger.info("In MODEL DIR: {}".format(scens))
                    for i in np.arange(len(scenarios[mipp][mip])):
                        if scenarios[mipp][mip][i] in scens:
                            m_dir = os.path.join(s_dir, scenarios[mipp][mip][i])
                            if os.path.isdir(m_dir) and os.listdir(m_dir):
                                logger.info('SCENARIO: {}'.format(scenarios[mipp][mip][i]))
                                run = [r for r in os.listdir(m_dir)]    
                                for rr in runs:
                                    if rr in run:
                                        logger.info('RUN: {}'.format(rr))
                                        j=0
                                        for dom in domains:
                                            logger.info('DOMAIN: {}'.format(dom))
                                            if dom in os.listdir(os.path.join(m_dir, rr)):
                                                v_dir = os.path.join(m_dir, rr, dom)
                                                var = [v for v in os.listdir(v_dir)]
                                                for vv in vars[j][dom]:
                                                    logger.info('VAR: {}'.format(vv))
                                                    if vv in var:
                                                        g_dir = os.path.join(v_dir, vv)
                                                        grids = [g for g in os.listdir(g_dir)]
                                                        for gg in grids:
                                                            logger.info('GRID: {}'.format(gg))
                                                            ve_dir = os.path.join(g_dir,gg)
                                                            vers = [ve for ve in os.listdir(ve_dir)]
                                                            for vee in vers:
                                                                logger.info('VERS: {}'.format(vee))
                                                                f_dir = os.path.join(ve_dir, vee)
                                                                if os.path.isdir(f_dir) and os.listdir(f_dir):
                                                                    [ofile_y, ofile_std] = data_crunch(
                                                                        f_dir, scenarios[mipp][mip][i], 
                                                                        vv, bandwidth, in_year, end_year)
                                                                    pimip_dir = os.path.join(path, 'CMIP')
                                                                    pimg_dir = os.path.join(pimip_dir, mg)
                                                                    if 'piControl' in os.listdir(os.path.join(pimg_dir, m)):
                                                                        pi_dir = os.path.join(pimg_dir, m, 'piControl', rr, dom, vv, gg)
                                                                        if os.path.isdir(pi_dir) and os.listdir(pi_dir): 
                                                                            if dom in os.listdir(os.path.join(pimg_dir, m, 'piControl', rr)):
                                                                                vers = [v for v in os.listdir(pi_dir)]
                                                                                vep = vers[0]
                                                                                fpi_dir = os.path.join(pi_dir, vep)
                                                                                if os.path.isdir(fpi_dir) and os.listdir(fpi_dir):
                                                                                    [ofile_piy, ofile_pistd] = data_crunch(
                                                                                        fpi_dir, 'piControl', vv, 
                                                                                        bandwidth, in_year, end_year)
                                                                                    #Computing tipping indicators
                                                                                    [time, lon, 
                                                                                     lat, data, 
                                                                                     indicators, masks] = tips(ofile_y, ofile_std, 
                                                                                                               ofile_piy, ofile_pistd, 
                                                                                                               vv, yrmaxchange)
                                                                                    #Mapping masks and indicators
                                                                                    lonm = np.array(lon)
                                                                                    latm = np.array(lat)
                                                                                    map(path_l, lon, lat, 
                                                                                        indicators[0], vv, vee, m, 
                                                                                        scenarios[mipp][mip][i], 'std', dom)
                                                                                    map(path_l, lon, lat, 
                                                                                        indicators[1], vv, vee, m, 
                                                                                        scenarios[mipp][mip][i], 'maxch', dom)
                                                                                    map(path_l, lon, lat, 
                                                                                        masks[0], vv, vee, m, 
                                                                                        scenarios[mipp][mip][i], 'mask_std', dom)
                                                                                    map(path_l, lon, lat, 
                                                                                        masks[1], vv, vee, m, 
                                                                                        scenarios[mipp][mip][i], 'mask_maxch', dom)
                                                                                    map(path_l, lon, lat, 
                                                                                        masks[2], vv, vee, m, 
                                                                                        scenarios[mipp][mip][i], 'mask_pc99', dom)
                                                                                    map(path_l, lon, lat,
                                                                                        masks[3], vv, vee, m, 
                                                                                        scenarios[mipp][mip][i], 'mask_combine', dom)
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[0]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8,
                                                                                                    )   
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_l, clusters, time,
                                                                                                    lon, lat, data, vv,
                                                                                                    vee, m, 'std', scenarios[mipp][mip][i],
                                                                                                    thres_gp, dom)
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[1]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8,
                                                                                                    )
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_l, clusters,
                                                                                                    time, lon, lat, data, vv,
                                                                                                    vee, m, 'maxch', scenarios[mipp][mip][i],
                                                                                                    thres_gp, dom)
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[2]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8,
                                                                                                    )
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_l, clusters,
                                                                                                    time, lon, lat, data, vv, 
                                                                                                    vee, m, 'pc_99', scenarios[mipp][mip][i],
                                                                                                    thres_gp, dom)
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[3]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8,
                                                                                                    )
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_l, clusters,
                                                                                                    time, lon, lat, data, vv,
                                                                                                    vee, m, 'combine', scenarios[mipp][mip][i],
                                                                                                    thres_gp, dom)
                                                                                    os.remove(ofile_piy)
                                                                                    os.remove(ofile_pistd)
                                                                                else:
                                                                                    logger.info('Directory is empty...')
                                                                        else:
                                                                            logger.info('The piControl run does not contain the requested variable.')
                                                                    else:
                                                                        logger.info('No piControl run found')
                                                                    os.remove(ofile_y)
                                                                    os.remove(ofile_std)
                                                                else:
                                                                    logger.info('The directory is empty...')
                                                    else:
                                                        logger.info("{} variable is not available".format(vv))
                                            else:
                                                logger.info('No variable in {} domain'.format(dom))
                                            j = j+1
                                    else:
                                        logger.info("Run {} is not available".format(rr))
                            else:
                                logger.info("Theere's nothing in this scenario.")
                        else:
                            logger.info("Experiment {} is not available".format(scenarios[mipp][mip][i]))
            else:
                logger.info("No model is found in {} model group".format(mg))
    else:
        logger.info("The MIP is empty")
    mipp = mipp+1
logger.info("Finished hunting for tipping. Now rest...")
