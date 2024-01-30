"""
WP5 in OptimESM: an algorithm for the detection of tipping elements in CMIP6 model runs

Created by:
Valerio Lembo (CNR-ISAC): v.lembo@isac.cnr.it

Versions:
10/5/23: And so it begins...
08/09/2023: Solved most issues with clustering, now polishing the script...
19/10/23: working version. Combines three masks: std, max jump, pc99
21/10/23: Added test on symmetry with K-S
16/11/23: Reorganized the header 
24/01/24: Printing out the time series to Netcdf
25/01/24: Appending historical time series to ssp scenarios
"""

from cdo import Cdo
from math import floor
Cdo.debug = True
from netCDF4 import Dataset as ds
import diptest
import optim_esm_tools as oet
from scipy import stats
# import rpy_symmetry as rsym
import datetime
import logging
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#Introductory section. 
cdo = Cdo()
now = datetime.datetime.now()
date = now.isoformat()
logfilen = 'log_hunt_{}.log'.format(date)
logging.basicConfig(filename=logfilen, level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

## User's options
# Fundamental parameters
bandwidth = 10      # for moving average, in years
yravg = bandwidth/2
yrmaxchange = 10        # Chunk length for maximum jumps evaluation
minnofyears = 11        # Shortest dataset to be considered
in_year = 2010          # Initial year for historical/ssp simulations
end_year = 2300         # Last year for ssp simulations
pc = [1, 5, 10, 25,     # set of percentiles for tail comparisons
      75, 90, 95, 99]
thres_gp = 150          # Minimal area (in gridpoints 1x1) for cluster retrieval

# Defining input/output paths, scenarios, variables and models to be analysed
path = '/work_big/datasets/synda/data/CMIP6/'
path_l = '/home/lembo/tipping_optimesm/figures_{}'.format(date)
try:
    os.makedirs(path_l)
except OSError:
    pass
project = ['CMIP', 'ScenarioMIP']
scenarios = [
    {'CMIP':[]},
    {'ScenarioMIP': ['ssp585', 'ssp370', 'ssp245', 'ssp126']}
    ]
runs = ['r1i1p1f1', 'r2i1p1f1']
model_groups = [
    # 'AS-RCEC', 'AWI',
    # 'BCC',
    # 'CAMS',
    # 'CCCma', 'CCCR-IITM', 
    # # 'CMCC', 
    # 'CSIRO', 
    # 'CSIRO-ARCCSS', 
    # 'E3SM-Project', 
    # 'EC-Earth-Consortium', 'FIO-QLNM',
    # 'HAMMOZ-Consortium', 
    # 'INM', 
    # 'IPSL', 
    # 'KIOST', 'MIROC', 'MOHC', 
    # 'MPI-M', 
    # 'MRI',
    # 'NASA-GISS', 
    # 'NCAR', 
    # 'NCC', 'NIMS-KMA',
    # 'NOAA-GFDL', 
    'NUIST', 
    'SNU',
    'THU'
    ]
vars = [
        # {'SImon':[]},
        {'SImon':['siconc']},
        {'Amon':['tas','pr']},
        # {'Omon':[]}]
        {'Omon':['sos','tos']}]
domains = ['SImon', 'Amon', 'Omon']

def data_crunch(f_dir,scen,var,filter,in_year,end_year):
    os.chdir(f_dir)
    tmp_dir = '/tmp/tmp_{}_{}'.format(scen,date)
    try:
        os.makedirs(tmp_dir)
    except OSError:
        pass
    ofile = os.path.join(tmp_dir, 'file_merged.nc')
    try:
        os.remove(ofile)
    except OSError:
        pass
    if len(os.listdir(f_dir))>1:
        # logger.info(len(os.listdir(f_dir)))
        cdo.mergetime(
            input=glob.glob(f_dir+'/{}_*.nc'.format(var)),
            options = '-O',
            output = ofile)
    else:
        ncfile = os.listdir(f_dir)
        shutil.copy(ncfile[0],ofile)
    ofile_y = os.path.join(tmp_dir, 'file_merged_y.nc')
    try:
        os.remove(ofile_y)
    except OSError:
        pass
    nyrs = cdo.nyear(input=ofile)[0]
    logger.info("Num. of years: {}".format(nyrs))
    print(nyrs)
    if int(nyrs)<10:
        logger.info('Not enough years')
        return
    else:
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
                options = '-P 8',
                output = ofile_y)
    else:
        if var=='siconc' or var=='sos' or var=='tos':
            # logger.info("Entering CDO manipulations...")
            cdo.setmisstoc(0,
                input = '-runmean,{} -selvar,{} -remapbil,r360x180 -yearmean {}'.format(filter,var,ofile),
                options = '-P 8',
                output = ofile_y)
        else:
            # logger.info("Entering CDO manipulations...")
            cdo.runmean(filter,
                input= '-remapbil,r360x180 -yearmean {}'.format(ofile),
                options = '-P 8',
                output = ofile_y)
    ofile_std = os.path.join(tmp_dir, 'file_merged_std.nc')
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
    var = data.variables[varname]
    unit = var.units 
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
    data = ds(filein_pi)
    time_pi = data.variables['time'][:]
    nyrs_pi = len(time_pi)
    finpi = nyrs_pi - yrmxch
    varpi = data.variables[varname][:,:,:]
    data = ds(filein_pistd)
    varpi_std = data.variables[varname][:,:,:]
    pcm_std = np.percentile(varpi_std, pc)
    var_shift = np.zeros(np.shape(var))
    var_shift[yrmxch:,:,:] = var[:fin,:,:]
    var_shiftdiff = var[yrmxch:,:,:]-var_shift[yrmxch:,:,:]
    var_diffabs = np.squeeze(np.abs(var_shiftdiff))
    var_timmax = np.nanmax(var_diffabs,0)
   
    oett = np.zeros(np.shape(var_timmax))
    oetd = np.zeros(np.shape(var_timmax))
    varn = np.where(var==0,np.nan,var)
    for ln in np.arange(len(lon)-1):
        for lt in np.arange(len(lat)-1):
            vart = np.squeeze(varn[:,lt,ln])
            maskn = np.where(~np.isnan(vart),1,0)
            if np.all(maskn):
                [stat, ps] = stats.kstest(vart-np.nanmean(vart), 'norm')
                [dipt, pd] = diptest.diptest(vart)
                oett[lt,ln] = ps
                oetd[lt,ln] = pd
                # oett[lt,ln] = rsym.p_symmetry(vart, test_statistic='KS')    
    varpi_shift = np.zeros(np.shape(varpi))
    varpi_shift[yrmxch:,:,:] = varpi[:finpi,:,:]
    varpi_shiftdiff = varpi[yrmxch:,:,:]-varpi_shift[yrmxch:,:,:]
    varpi_diffabs = np.squeeze(np.abs(varpi_shiftdiff))
    varpi_timmax = np.nanmax(varpi_diffabs,0)
    pcm_jump = np.percentile(varpi_timmax, pc)
    
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
    # mask_75p = np.where(
    #     ((var>pcm[4,:,:]).astype(bool) |
    #      (var<pcm[0,:,:]).astype(bool) &
    #     (var!=0.).astype(bool) &
    #     (~np.isnan(var))),
    #     1, 0)
    # mask_90p = np.where(
    #     ((var>pcm[5,:,:]).astype(bool) |
    #     (var<pcm[1,:,:]).astype(bool) &
    #     (var!=0.).astype(bool) &
    #     (~np.isnan(var))),
    #     1, 0)
    # mask_95p = np.where(
    #     ((var>pcm[6,:,:]).astype(bool) |
    #     (var<pcm[2,:,:]).astype(bool) &
    #     (var!=0.).astype(bool) &
    #     (~np.isnan(var))),
    #     1, 0)
    mask_99p = np.where(
        ((var>pcm[7,:,:]).astype(bool) |
        (var<pcm[3,:,:]).astype(bool) &
        (var!=0.).astype(bool) &
        (~np.isnan(var))),
        1, 0)
    mask_99ps = np.where(
        ((np.nansum(mask_99p[-30:,:,:].astype(int),axis=0)/30)==1.).astype(bool),
        1, 0)
    mask_ps = np.where(((oett>(pc[2]/100)).astype(bool) &
                         (oett!=0.).astype(bool) &
                         (~np.isnan(oett))),
                         1, 0)
    mask_pd = np.where(((oetd<(pc[2]/100)).astype(bool) &
                         (oetd!=0.).astype(bool) &
                         (~np.isnan(oetd))),
                         1, 0)
    mask_combine = np.where(
        ((mask_std.astype(int) + 
          mask_max.astype(int) +
          mask_ps.astype(int) +
          np.nansum(mask_99p[-30:,:,:].astype(int),axis=0)/30)>=3).astype(bool),
        1, 0)
    indicators = [var_std, var_timmax, oett, oetd]
    masks = [mask_std, mask_max, mask_99ps, mask_combine, mask_ps, mask_pd]
    return time, lon, lat, unit, var, indicators, masks


def plotting_clusters(path, file_in, file_pin, file_hin, clusters, time, lon, lat, unit, data, ind, 
                      vv, vee, mod, method, scen, thres):
    for cl in range(len(clusters[:,0,0])):
        cl_tser = clusters[cl,:,:]
        if np.nansum(cl_tser)>thres:
            logger.info("Cluster {} of {} for criterion {} passes the threshold for the area".format(cl, 
                                                                                                     np.size(clusters,axis=0), 
                                                                                                     method))
            crate = np.nansum(np.nansum(cl_tser,axis=0))/(len(lon)*len(lat))
            
            data = ds(file_in)
            var = data.variables[vv]
            fld = cl_tser[np.newaxis,:,:] * var
            tser = np.nanmean(np.nanmean(fld,axis=2),axis=1)/crate
            data = ds(file_pin)
            var = data.variables[vv]
            fld_pi = cl_tser[np.newaxis,:,:] * var
            tser_pi = np.nanmean(np.nanmean(fld_pi,axis=2),axis=1)/crate
            data = ds(file_hin)
            var = data.variables[vv]
            fld_hi = cl_tser[np.newaxis,:,:] * var
            tser_hi = np.nanmean(np.nanmean(fld_hi,axis=2),axis=1)/crate
            
            std = cl_tser * ind[0]
            std_m = np.nanmean(np.where(std==0,np.nan,std))
            mjump = cl_tser * ind[1]
            mjump_m = np.nanmean(np.where(mjump==0,np.nan,mjump)) 
            # dipt = cl_tser * ind[3]
            fld_n = np.where(fld==0,np.nan,fld)
            tstd = np.nanstd(np.nanstd(fld_n,axis=2),axis=1)
            
            
            [dipt,pd] = diptest.diptest(tser) 
            [stat, ps] = stats.kstest(tser-np.nanmean(tser), 'norm')
            plt.figure(figsize=(12, 6))  # Set the figure size
            ax1 = plt.subplot(2,2,1)
            map_inlet(ax1, lon, lat, np.squeeze(clusters[cl,:,:]))
            ax2 = plt.subplot(2,2,2)
            plot_tser_inlet(time, tser, vv)
            ax3 = plt.subplot(2,2,3)
            plot_tser_inlet(time, tstd, 'std')
            ax4 = plt.subplot(2,2,4)
            tab = plt.table(cellText = [['St. dev.','Max. Jump'],
                                        [str(std_m), str(mjump_m)],
                                        ['Dip pval', 'KS pval'],
                                        [str(pd), str(ps)]],
                            # colLabels = [],
                            loc='center')
            plt.axis('off')
            plt.suptitle(vv + " " + mod + " " + vee + " " + scen + " " + method)
            file_f = path + "/" + vv + "_" + mod + "_" + vee + "_" + scen + "_cl_" + method + "_" + str(cl) + ".png"
            plt.savefig(file_f)
            plt.close()
            
            file = path + '/ssp.nc'
            pr_output(tser, vv, method, file_in, file, unit)
            file_hi = path + '/hist.nc'
            pr_output(tser_hi, vv, method, file_hin, file_hi, unit)
            file_pi = path + "/" + vv + "_" + mod + "_" + vee + "_" + scen + "_cl_" + method + "_" + str(cl) + "_piC.nc" 
            pr_output(tser_pi, vv, method, file_pin, file_pi, unit)
            file_out = path + "/" + vv + "_" + mod + "_" + vee + "_" + scen + "_cl_" + method + "_" + str(cl) + ".nc"    
            cdo.mergetime(input='{} {}'.format(file_hi,file),
                          output = file_out)
            os.remove(file)
            os.remove(file_hi)
        else:
            pass


def varatts(w_nc_var, method, tres, unit):
    """Add attibutes to the variables, depending on name and time res.

    Arguments:
    - w_nc_var: a variable object;
    - varname: the name of the variable, among ta, ua, va and wap;
    - tres: the time resolution (daily or annual);

    @author: Chris Slocum (2014), modified by Valerio Lembo (2018).
    """
    if tres == 0:
        tatt = "Daily\nM"
    elif tres == 1:
        tatt = "Annual mean\nM"
    if method == 'std':
        w_nc_var.setncatts({
            'long_name': "Standard Deviation",
            'units': unit,
            'var_desc': "Time series of field averaged on cluster identified by standard deviation",
            'statistic': tatt
        })
    elif method == 'maxch':
        w_nc_var.setncatts({
            'long_name': "Max. 10yr jump",
            'units': unit,
            'var_desc': "Time series of field averaged on cluster identified by max 10-yr jump",
            'statistic': tatt
        })
    elif method == 'pc_99':
        w_nc_var.setncatts({
            'long_name': "Last 30yr 99-percentile",
            'units': unit,
            'var_desc': "Time series of field averaged on cluster identified by 99-pc method",
            'statistic': tatt
        })
    elif method == 'combine':
        w_nc_var.setncatts({
            'long_name': "Combined indices",
            'units': unit,
            'var_desc': "Time series of field averaged on cluster identified by combined method",
            'statistic': tatt
        })
        

def pr_output(varo, varname, method, filep, nc_f, unit):
    """Print outputs to NetCDF.

    Save fields to NetCDF, retrieving information from an existing
    NetCDF file. Metadata are transferred from the existing file to the
    new one.
    Arguments:
    - varo: the field to be stored;
    - varname: the name of the variables to be saved;
    - filep: the existing dataset, containing the metadata;
    - nc_f: the name of the output file;

    PROGRAMMER(S)
        Chris Slocum (2014), modified by Valerio Lembo (2018).
    """
    with ds(nc_f, 'w', format='NETCDF4') as w_nc_fid:
        # w_nc_fid.description = "Outputs of LEC program"
        with ds(filep, 'r') as nc_fid:
            # Extract data from NetCDF file
            time = nc_fid.variables['time'][:]
            nyrs = int(len(time))
            # Writing NetCDF files
            w_nc_fid.createDimension('time', nyrs)
            w_nc_dim = w_nc_fid.createVariable(
                'time', nc_fid.variables['time'].dtype, ('time', ))
            for ncattr in nc_fid.variables['time'].ncattrs():
                w_nc_dim.setncattr(
                    ncattr, nc_fid.variables['time'].getncattr(ncattr))
        w_nc_fid.variables['time'][:] = time[0:nyrs]
        w_nc_var = w_nc_fid.createVariable(varname, 'f8', ('time'))
        varatts(w_nc_var, method, 1, unit)
        w_nc_fid.variables[varname][:] = varo
                                                           

def plot_tser(path, time, var, vname, ver, model, scen, name):
    # Plotting
    yrs = julian_date_to_decimal_years(time)
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(yrs, var, color='blue', linewidth=1)
    plt.title(vname + " " + model + " " + ver + " " + scen)
    plt.xlabel('Years')  # Set the x-axis label
    plt.ylabel(vname)  # Set the y-axis label
    plt.grid(True)  # Enable gridlines
    plt.tight_layout()  # Adjust the spacing of the plot
    plt.savefig(path + "/" + model + "/" + vname + "_" + model + "_" + ver + "_" + scen + "_" + name + "_tser.png")
    plt.close()
    
def plot_tser_inlet(time, var, vname):
    # Plotting
    yrs = julian_date_to_decimal_years(time)
    # plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(yrs, var, color='blue', linewidth=1)
    plt.xlabel('Years')  # Set the x-axis label
    plt.ylabel(vname)  # Set the y-axis label
    plt.grid(True)  # Enable gridlines
    plt.tight_layout()  # Adjust the spacing of the plot

def map(path, lons, lats, data, var, ver, model, scen, mode):
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
    plt.title(var + " " + model + " " + ver + " " + scen + "\n " + mode)
    plt.colorbar()
    # show and save the map
    plt.savefig(path + "/" + var + "_" + model + "_" + ver + "_" + scen + "_" + mode + ".png")
    plt.close()
    
def map_inlet(ax, lons, lats, data):
    m = Basemap(projection='cyl', resolution='c', lon_0=180., ax=ax)
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
    # plt.colorbar(cax=ax)


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
                    # for i in np.arange(len(scenarios[mipp][mip])):
                    for ss in scenarios[mipp][mip]:
                        # if scenarios[mipp][mip][i] in scens:
                        if ss in scens:
                            m_dir = os.path.join(s_dir, ss)
                            if os.path.isdir(m_dir) and os.listdir(m_dir):
                                logger.info('SCENARIO: {}'.format(ss))
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
                                                                    try:
                                                                        [ofile_y, ofile_std] = data_crunch(
                                                                            f_dir, ss, 
                                                                            vv, bandwidth, in_year, end_year)
                                                                    except TypeError:
                                                                        continue
                                                                    cmip_dir = os.path.join(path, 'CMIP')
                                                                    cmg_dir = os.path.join(cmip_dir, mg)
                                                                    if 'piControl' in os.listdir(os.path.join(cmg_dir, m)):
                                                                        # logger.info("The piControl is present...")
                                                                        pi_dir = os.path.join(cmg_dir, m, 'piControl', rr)
                                                                        hi_dir = os.path.join(cmg_dir, m, 'historical', rr) 
                                                                        piv_dir = os.path.join(pi_dir, dom, vv, gg)
                                                                        hiv_dir = os.path.join(hi_dir, dom, vv, gg)
                                                                        if os.path.isdir(piv_dir) and os.listdir(piv_dir) and os.path.isdir(hiv_dir) and os.listdir(hiv_dir):   
                                                                            if dom in os.listdir(pi_dir) and dom in os.listdir(hi_dir):
                                                                                vers = [v for v in os.listdir(piv_dir)]
                                                                                vep = vers[0]
                                                                                fpi_dir = os.path.join(piv_dir, vep)
                                                                                vers = [v for v in os.listdir(hiv_dir)]
                                                                                veh = vers[0]
                                                                                fhi_dir = os.path.join(hiv_dir, veh)
                                                                                if os.path.isdir(fpi_dir) and os.listdir(fpi_dir):
                                                                                    try:
                                                                                        [ofile_piy, ofile_pistd] = data_crunch(
                                                                                            fpi_dir, 'piControl', vv, 
                                                                                            bandwidth, in_year, end_year)
                                                                                    except TypeError:
                                                                                        continue
                                                                                    #Computing tipping indicators
                                                                                    try:
                                                                                        [time, lon, 
                                                                                        lat, unit, data, 
                                                                                        indicators, masks] = tips(ofile_y, ofile_std, 
                                                                                                                   ofile_piy, ofile_pistd, 
                                                                                                                   vv, yrmaxchange)
                                                                                    except TypeError as e:
                                                                                        continue
                                                                                    #Mapping masks and indicators
                                                                                    try:
                                                                                        [ofile_hiy, ofile_histd] = data_crunch(
                                                                                            fhi_dir, 'historical', vv, 
                                                                                            bandwidth, 1850, 2014)
                                                                                    except TypeError:
                                                                                        continue
                                                                                    lonm = np.array(lon)
                                                                                    latm = np.array(lat)
                                                                                    path_f = '{}/{}/{}/{}'.format(path_l, m, ss, vv) 
                                                                                    try:
                                                                                        os.makedirs(path_f)
                                                                                    except OSError:
                                                                                      pass
                                                                                    map(path_f, lon, lat, 
                                                                                        indicators[0], vv, vee, m, 
                                                                                        ss, 'std')
                                                                                    map(path_f, lon, lat, 
                                                                                        indicators[1], vv, vee, m, 
                                                                                        ss, 'maxch')
                                                                                    map(path_f, lon, lat, 
                                                                                        indicators[2], vv, vee, m, 
                                                                                        ss, 'sym')
                                                                                    map(path_f, lon, lat, 
                                                                                        indicators[1], vv, vee, m, 
                                                                                        ss, 'mmod')
                                                                                    map(path_f, lon, lat, 
                                                                                        masks[0], vv, vee, m, 
                                                                                        ss, 'mask_std')
                                                                                    map(path_f, lon, lat, 
                                                                                        masks[1], vv, vee, m, 
                                                                                        ss, 'mask_maxch')
                                                                                    map(path_f, lon, lat, 
                                                                                        masks[2], vv, vee, m, 
                                                                                        ss, 'mask_pc99')
                                                                                    map(path_f, lon, lat,
                                                                                        masks[3], vv, vee, m, 
                                                                                        ss, 'mask_combine')
                                                                                    map(path_f, lon, lat,
                                                                                        masks[4], vv, vee, m, 
                                                                                        ss, 'mask_sym')
                                                                                    map(path_f, lon, lat,
                                                                                        masks[5], vv, vee, m, 
                                                                                        ss, 'mask_mmod')
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[0]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8)
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_f, ofile_y, ofile_piy, ofile_hiy, 
                                                                                                    clusters, time, lon, lat, unit, data, 
                                                                                                    indicators, vv, vee, m, 'std', ss,
                                                                                                    thres_gp)
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[1]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8,
                                                                                                    )
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_f, ofile_y, ofile_piy, ofile_hiy,
                                                                                                    clusters, time, lon, lat, unit, 
                                                                                                    data, indicators, vv, vee, m, 
                                                                                                    'maxch', ss, thres_gp)
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[2]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8,
                                                                                                    )
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_f, ofile_y, ofile_piy, ofile_hiy, 
                                                                                                    clusters, time, lon, lat, unit, data, 
                                                                                                    indicators, vv, vee, m, 'pc_99', ss,
                                                                                                    thres_gp)
                                                                                    clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                                    np.array(np.bool_(np.squeeze(masks[3]))),
                                                                                                    latm,
                                                                                                    lonm,
                                                                                                    max_distance_km='infer',
                                                                                                    min_samples=8)
                                                                                    if len(clusters) >= 1:
                                                                                        clusters = np.array(clusters, dtype=int)
                                                                                        plotting_clusters(path_f, ofile_y, ofile_piy, ofile_hiy, 
                                                                                                    clusters, time, lon, lat, unit, data, 
                                                                                                    indicators, vv, vee, m, 'combine', ss,
                                                                                                    thres_gp)
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
                            logger.info("Experiment {} is not available".format(ss))
            else:
                logger.info("No model is found in {} model group".format(mg))
    else:
        logger.info("The MIP is empty")
    mipp = mipp+1
logger.info("Finished hunting for tipping. Now rest...")
