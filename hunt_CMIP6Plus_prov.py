"""
WP5 in OptimESM: an algorithm for the detection of tipping elements in CMIP6 model runs

NB: This is a version for pre-published CMORized OPTIMESM runs.

Created by:
Valerio Lembo (CNR-ISAC): v.lembo@isac.cnr.it

Versions:
05/12/25: Cloned original project to CINECA g100
11/03/26: Revisited the order of the paths

"""

from cdo import Cdo
from math import floor
Cdo.debug = True
from netCDF4 import Dataset as ds
import diptest
import optim_esm_tools.optim_esm_tools as oet
print(oet.__file__)
print(hasattr(oet, "analyze"))
from scipy import stats
from scipy.signal import find_peaks
# import rpy_symmetry as rsym
import datetime
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from mpl_toolkits.basemap import Basemap

#Introductory section. 
cdo = Cdo()
now = datetime.datetime.now()
today = now.isoformat()
logfilen = 'log_hunt_onlysiconc_{}.log'.format(today)
logging.basicConfig(filename=logfilen, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#Script version
hunt_vers = 'v2.0, 11/03/2025'

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
y30_pcmin = 0
y30_pcmax = 7
std_pcmax = 7
maxj_pcmax = 7
oett_pcmax = 1 
oetd_pcmax = 1
thres_gp = 150           # Minimal area (in gridpoints 1x1) for cluster retrieval
plev= 50000

# Defining input/output paths, scenarios, variables and models to be analysed
path = '/g100_store/DRES_OptimESM/ESGF/prepub/'
tmp_path = '/g100_scratch/userexternal/vlembo00/tmp/'
path_l = '/g100_work/IscrC_SPREM3-5/vlembo00/optimesm-tipping/maxj99_std99_30p99_oetd95_oett95/figures_{}_gwl4_TIPMIP'.format(today)

try:
    os.makedirs(path_l)
except OSError:
    pass
project = ['CMIP', 'TIPMIP', 'ScenarioMIP']
scenarios = {
    #'CMIP': [
    #    'esm-up2p0', 'esm-up2p0-gwl1p5', 'esm-up2p0-gwl2p0', 'esm-up2p0-gwl3p0', 'esm-up2p0-gwl4p0',
    #    'esm-up2p0-gwl5p0', 'esm-up2p0-gwl6p0', 'esm-up2p0-gwl2p0-50y-dn1p0', 'esm-up2p0-gwl2p0-50y-dn2p0', 
    #    'esm-up2p0-gwl4p0-200y-dn2p0', 'esm-up2p0-gwl1p5-50y-dn2p0', 'esm-up2p0-gwl4p0-50y-dn1p0'],
    #'TIPMIP': [
    #    'esm-up2p0', 'esm-up2p0-gwl1p5', 'esm-up2p0-gwl2p0', 'esm-up2p0-gwl3p0', 'esm-up2p0-gwl4p0',
    #    'esm-up2p0-gwl5p0', 'esm-up2p0-gwl6p0', 'esm-up2p0-gwl2p0-50y-dn1p0', 'esm-up2p0-gwl2p0-50y-dn2p0', 
    #    'esm-up2p0-gwl4p0-200y-dn2p0', 'esm-up2p0-gwl1p5-50y-dn2p0', 'esm-up2p0-gwl4p0-50y-dn1p0'],
    'CMIP': ['esm-up2p0-gwl2p0', 'esm-up2p0-gwl4p0'],
    'TIPMIP': ['esm-up2p0-gwl2p0', 'esm-up2p0-gwl4p0'],
    #{'CMIP':['esm-up2p0-gwl1p5']},
    'ScenarioMIP': []
    # {'ScenarioMIP': ['ssp585']}
}

runs = ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f2', 'r1i1p3f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1']
folders = [
        # 'bsc',
        # 'cnrm',
        # 'dmi'
        'ipsl',
        # 'knmi',
        'smhi',
        'mohc'
        ]
model_groups = [
    'AWI',
    # 'CMCC', 
    'EC-Earth-Consortium',
    'CNRM-CERFACS',
    'IPSL',
    'MOHC'
    ]
vars = [
        # {'day':['va']},
        #{'Lmon': ['mrro', 'mrso']},
        #{'LPmon': ['mrro', 'mrso']},
        {'SImon': ['siconc']},
        #{'Amon': ['tas', 'pr']},
        #{'APmon': ['tas', 'pr']},
        #{'Omon': ['sos', 'tos']},
        #{'OPmon': ['sos', 'tos']}
        {'Lmon':[]},
        {'LPmon':[]},
        # {'SImon':[]},
        {'Amon':[]},
        {'APmon':[]},
        {'Omon':[]},
        {'OPmon':[]}
        ]
domains = ['Lmon', 'LPmon', 'SImon', 'Amon', 'APmon', 'Omon', 'OPmon']

def data_crunch(f_dir,scen,var,filter,in_year,end_year):
    os.chdir(f_dir)
    tmp_dir = '{}tmp_{}_{}'.format(tmp_path,scen,today)
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
    ofile_my = os.path.join(tmp_dir, 'file_merged_my.nc')
    try:
        os.remove(ofile_my)
    except OSError:
        pass
    ofile_y = os.path.join(tmp_dir, 'file_merged_y.nc')
    try:
        os.remove(ofile_y)
    except OSError:
        pass
    nyrs = cdo.nyear(input=ofile)[0]
    logger.info("Num. of years: {}".format(nyrs))
    print(nyrs)
    if int(nyrs)<yrmaxchange:
        logger.info('Not enough years')
        return
    else:
        pass
    if scen=='ssp585' or scen=='ssp245' or scen=='ssp126':
        if var=='siconc' or var=='sos' or var=='tos':
            cdo.setmisstoc(0,
                input= '-selyear,{}/{} -runmean,{} -selvar,{} -remapbil,r360x180 -yearmean {}'.format(in_year,end_year,filter,var,ofile),
                options = '-P 8',
                output = ofile_my)
            cdo.setmisstoc(0,
                input= '-selyear,{}/{} -selvar,{} -remapbil,r360x180 -yearmean {}'.format(in_year,end_year,var,ofile),
                options = '-P 8',
                output = ofile_y)
        elif var=='zg' or var=='ta' or var=='ua' or var=='va':
            cdo.remapbil('r360x180',
                input= '-selyear,{}/{} -yearmean -sellevel,{} {}'.format(in_year,end_year,plev,ofile),
                options = '-P 8',
                output = ofile_y)
            cdo.runmean(filter,
                input= ofile_y,
                options = '-P 8',
                output = ofile_my)
        else:
            cdo.runmean(filter,
                input= '-selyear,{}/{} -remapbil,r360x180 -yearmean {}'.format(in_year,end_year,ofile),
                options = '-P 8',
                output = ofile_my)
            cdo.remapbil('r360x180',
                input= '-selyear,{}/{} -yearmean {}'.format(in_year,end_year,ofile),
                options = '-P 8',
                output = ofile_y)
    else:
        if var=='siconc' or var=='sos' or var=='tos':
            # logger.info("Entering CDO manipulations...")
            cdo.setmisstoc(0,
                input = '-runmean,{} -selvar,{} -remapbil,r360x180 -yearmean {}'.format(filter,var,ofile),
                options = '-P 8',
                output = ofile_my)
            cdo.setmisstoc(0,
                input = '-selvar,{} -remapbil,r360x180 -yearmean {}'.format(var,ofile),
                options = '-P 8',
                output = ofile_y)
        elif var=='zg' or var=='ta' or var=='ua' or var=='va':
            cdo.runmean(filter,
                input= '-remapbil,r360x180 -yearmean -sellevel,{} {}'.format(plev,ofile),
                options = '-P 8',
                output = ofile_my)
            cdo.remapbil('r360x180',
                input= '-yearmean -sellevel,{} {}'.format(plev,ofile),
                options = '-P 8',
                output = ofile_y)
        else:
            # logger.info("Entering CDO manipulations...")
            cdo.runmean(filter,
                input= '-remapbil,r360x180 -yearmean {}'.format(ofile),
                options = '-P 8',
                output = ofile_my)
            cdo.remapbil('r360x180',
                input= '-yearmean {}'.format(ofile),
                options = '-P 8',
                output = ofile_y)
    ofile_std = os.path.join(tmp_dir, 'file_merged_std.nc')
    try:
        os.remove(ofile_std)
    except OSError:
        pass                                                    
    cdo.timstd(
        input = '-detrend {}'.format(ofile_my),
        output = ofile_std)
    try:
        os.remove(ofile)
    except OSError:
        pass 
    return ofile_y, ofile_my, ofile_std


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
    var = np.squeeze(data.variables[varname])
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
    var_std = np.squeeze(data.variables[varname])
    data = ds(filein_pi)
    time_pi = data.variables['time'][:]
    nyrs_pi = len(time_pi)
    finpi = nyrs_pi - yrmxch
    varpi = np.squeeze(data.variables[varname])
    data = ds(filein_pistd)
    varpi_std = np.squeeze(data.variables[varname][:,:,:])
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
        ((np.abs(var_std)>pcm_std[std_pcmax]).astype(bool) &
        ((np.abs(varpi_std)!=0.).astype(bool)) &
        (~np.isnan(varpi_std))), 
        1, 0)
    mask_max = np.where(
        ((np.abs(var_timmax)>pcm_jump[maxj_pcmax]).astype(bool) & 
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
        ((var>pcm[y30_pcmax,:,:]).astype(bool) |
        (var<pcm[y30_pcmin,:,:]).astype(bool) &
        (var!=0.).astype(bool) &
        (~np.isnan(var))),
        1, 0)
    mask_99ps = np.where(
        ((np.nansum(mask_99p[-30:, :, :].astype(int),
                    axis=0)/30) == 1.).astype(bool),
        1, 0)
    mask_ps = np.where(((oett>(pc[oett_pcmax]/100)).astype(bool) &
                         (oett!=0.).astype(bool) &
                         (~np.isnan(oett))),
                         1, 0)
    mask_pd = np.where(((oetd<(pc[oetd_pcmax]/100)).astype(bool) &
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


def plotting_clusters(path, file_in, file_pin, file_hin, file_min, file_mpin,
                      file_mhin, clusters, time, lon, lat, unit, data, ind,
                      vv, vee, mod, method, scen, thres):
    clusters = np.swapaxes(clusters, 1, 2)
    for cl in range(len(clusters[:, 0, 0])):
        cl_tser = clusters[cl, :, :]
        if np.nansum(cl_tser) > thres:
            logger.info("Cluster {} of {} for criterion {} passes the threshold for the area".format(cl, 
                                                                                                     np.size(clusters,axis=0), 
                                                                                                     method))
            crate = np.nansum(np.nansum(cl_tser, axis=0))/(len(lon)*len(lat))

            data = ds(file_min)
            var = data.variables[vv]
            time = data.variables['time'][:]
            nyrs = len(time)
            fin = nyrs - yrmaxchange
            fld = cl_tser[np.newaxis, :, :] * var
            tser = np.nanmean(np.nanmean(fld, axis=2), axis=1)/crate
            yrs = julian_date_to_decimal_years(time)

            data = ds(file_mpin)
            var = data.variables[vv]
            time_pi = data.variables['time'][:]

            fld_pi = cl_tser[np.newaxis, :, :] * var
            nyrs_pi = len(time_pi)
            finpi = nyrs_pi - yrmaxchange
            data = ds(file_mhin)
            var = data.variables[vv]

            fld_hi = cl_tser[np.newaxis, :, :] * var
            tser_hi = np.nanmean(np.nanmean(fld_hi, axis=2), axis=1)/crate

            tser_all = np.append(tser_hi, tser)
            time_all = np.linspace(1850, 1850+len(tser_all), len(tser_all))
            yrs_all = time_all.astype(int)
            # yrs_all = julian_date_to_decimal_years(time_all)

            # Print cluster mean time mean values to table
            std = cl_tser * ind[0]
            std_m = np.nanmean(np.where(std == 0, np.nan, std))
            mjump = cl_tser * ind[1]
            mjump_m = np.nanmean(np.where(mjump == 0, np.nan, mjump))

            fld_n = np.where(fld == 0, np.nan, fld)
            tstd = np.nanstd(np.nanstd(fld_n, axis=2), axis=1)
            fld_npi = np.where(fld_pi == 0, np.nan, fld_pi)
            std_pim = np.nanstd(np.nanstd(fld_npi, axis=2), axis=1)
            pcm_std = np.percentile(std_pim, pc)

            peaks_idx, _ = find_peaks(tstd, distance=10)
            top5_std_idx = peaks_idx[np.argsort(tstd[peaks_idx])[-10:]]
            top5_std_mask = np.zeros(len(tstd), dtype=bool)
            top5_std_mask[top5_std_idx] = True
            # threshold_5 = np.min(tstd[top5_idx])
            # threshold_5 = np.sort(tstd)[-5]

            fld_npim = np.squeeze(np.nanmean(np.nanmean(fld_npi,
                                                        axis=2),
                                             axis=1))
            fld_npi_shift = np.zeros(np.shape(fld_npim))
            fld_npi_shift[yrmaxchange:] = fld_npim[:finpi]
            mjump_npi = np.squeeze(np.abs(fld_npim[yrmaxchange:] -
                                          fld_npi_shift[yrmaxchange:]))
            mjump_pi = np.nanmax(mjump_npi, 0)
            # mjump_pim = np.nanmean(np.where(mjump_pi == 0, np.nan, mjump_pi))
            # pcm_jump = np.percentile(mjump_pim, pc)
            fld_nm = np.squeeze(np.nanmean(np.nanmean(fld_n,
                                                      axis=2),
                                           axis=1))
            fld_n_shift = np.zeros(np.shape(fld_nm))
            fld_n_shift[yrmaxchange:] = fld_nm[:fin]
            fld_n_shiftdiff = fld_nm[yrmaxchange:]-fld_n_shift[yrmaxchange:]
            fld_n_diffabs = np.squeeze(np.abs(fld_n_shiftdiff))

            peaks_idx, _ = find_peaks(fld_n_diffabs, distance=10)
            top5_maxj_idx = peaks_idx[np.argsort(fld_n_diffabs[peaks_idx])[-10:]]
            top5_maxj_mask = np.zeros(len(fld_n_diffabs), dtype=bool)
            top5_maxj_mask[top5_maxj_idx] = True
            tolerance = 25  # +/- neighbours
            common_idx = []
            common_std_idx = []
            top5_maxj_idx_shift = top5_maxj_idx - yrmaxchange
            for idx_j in top5_maxj_idx_shift:
                for idx_s in top5_std_idx:
                    if np.abs(idx_j - idx_s) <= tolerance:
                        common_idx.append(idx_j)
                        common_std_idx.append(idx_s)
                        break  # avoid duplicates
            common_idx = np.array(common_idx)
            if len(common_idx) > 0:
                top5_maxj_mask[:] = False
                top5_maxj_mask[common_idx+yrmaxchange] = True
                top5_std_mask[:] = False
                top5_std_mask[common_std_idx] = True
            flag_mjump = np.where(((np.abs(fld_n_diffabs) > mjump_pi)
                                   &
                                   top5_maxj_mask).astype(bool),
                                  1, 0)
            flag_mjump = np.atleast_1d(flag_mjump)
            flag_std = np.where(((np.abs(tstd) > pcm_std[std_pcmax]) &
                                top5_std_mask).astype(bool),
                                1, 0)
            flag_std = np.atleast_1d(flag_std)

            [dipt, pd] = diptest.diptest(tser)
            [stat, ps] = stats.kstest(tser-np.nanmean(tser), 'norm')
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            # plt.figure(figsize=(12, 8))  # Set the figure size
            ax1 = axes[0, 0]
            # ax1 = plt.subplot(2,2,1)
            map_inlet(ax1, lon, lat, np.squeeze(clusters[cl, :, :]))
            ax1.axis('off')
            ax2 = axes[0, 1]
            # ax2 = plt.subplot(2,2,2)
            plot_tser_inlet(ax2, yrs, tser, vv, flag_std, flag_mjump)
            ax3 = axes[1, 0]
            # ax3 = plt.subplot(2,2,3)
            plot_tser_inlet(ax3, yrs, tstd, 'std', flag_std, flag_mjump)
            ax4 = axes[1, 1]
            # Plot maps of field differences around the MNS
            diff = np.zeros([len(common_idx),
                            len(lat), len(lon)])
            d=0
            for c in common_idx:
                diff[d, :, :] = (np.nanmean(fld_n[c:c+10, :, :], 0) -
                                 np.nanmean(fld_n[c-10:c, :, :], 0))
                d = d+1
            diffm = np.nanmean(diff, axis=0)
            rr, cc = np.where(~np.isnan(diffm))
            print(rr)
            print(cc)
            map_inlet_fld(ax4, lon[min(cc):max(cc)],
                          lat[min(rr):max(rr)],
                          np.squeeze(diffm[min(rr):max(rr), min(cc):max(cc)]))
            # ax4 = plt.subplot(2,2,4)
            # tab = ax4.table(cellText = [['St. dev.','Max. Jump'],
            #                             [str(std_m), str(mjump_m)],
            #                             ['Dip pval', 'KS pval'],
            #                             [str(pd), str(ps)]],
            #                 # colLabels = [],
            #                 loc='center')
            ax4.axis('off')
            fig.suptitle(vv + " " + mod + " " + vee + " " + scen + " " + method + "Cluster " + str(cl))
            file_f = path + "/" + vv + "_" + mod + "_" + vee + "_" + scen + "_cl_" + method + "_" + str(cl) + ".png"
            plt.savefig(file_f)
            plt.close()

            data = ds(file_in)
            var = data.variables[vv]
            fld = cl_tser[np.newaxis, :, :] * var
            tser = np.nanmean(np.nanmean(fld, axis=2), axis=1)/crate
            data = ds(file_pin)
            var = data.variables[vv]
            fld_pi = cl_tser[np.newaxis, :, :] * var
            tser_pi = np.nanmean(np.nanmean(fld_pi, axis=2), axis=1)/crate
            data = ds(file_hin)
            var = data.variables[vv]
            fld_hi = cl_tser[np.newaxis, :, :] * var
            tser_hi = np.nanmean(np.nanmean(fld_hi, axis=2), axis=1)/crate

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


def plot_tser_inlet(ax, time, var, vname, flag1, flag2):
    # Plotting
    # plt.figure(figsize=(10, 6))  # Set the figure size
    ax.plot(time, var, color='blue', linewidth=1)
    if flag1.any():
        flag1_indices = np.where(flag1 == 1)[0]
        for i, idx in enumerate(flag1_indices):
            ax.axvline(x=time[idx], color='red',
                       linestyle='--', linewidth=0.8,
                       label='' if i == 0 else None)
        # first_idx = np.argmax(flag1 == 1)
        # ax.axvline(x=time[first_idx], color='red', linestyle='--', linewidth=1.5)
    if flag2.any():
        flag2_indices = np.where(flag2 == 1)[0]
        for i, idx in enumerate(flag2_indices):
            ax.axvline(x=time[idx], color='blue',
                       linestyle='--', linewidth=0.8,
                       label='' if i == 0 else None)
        # first_idx = np.argmax(flag2 == 1)
        # ax.axvline(x=time[first_idx], color='blue', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Years')  # Set the x-axis label
    ax.set_ylabel(vname)  # Set the y-axis label
    ax.grid(True)  # Enable gridlines
    # plt.tight_layout()  # Adjust the spacing of the plot

def map(path, lons, lats, data, var, ver, model, scen, mode):
    ax = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    # draw map features
    ax.coastlines(resolution='110m')
    # m.drawcoastlines()
    # m.drawcountries()
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    # Fill continents with color (equivalent to m.fillcontinents(color='coral'))
    ax.add_feature(cfeature.LAND, facecolor='coral')
    # Fill ocean/lakes with color (equivalent to lake_color='aqua')
    ax.add_feature(cfeature.OCEAN, facecolor='aqua')
    # m.fillcontinents(color='coral', lake_color='aqua')
    # plot data on the map
    vmax = np.nanmax(np.abs(data))
    vmin = 0.
    ax.pcolormesh(lons, lats, np.squeeze(data), cmap='Reds', vmin = vmin, vmax = vmax)
    # m.pcolormesh(lons, lats, np.squeeze(data), cmap='Reds', vmin = vmin, vmax = vmax)
    # add title
    plt.title(var + " " + model + " " + ver + " " + scen + "\n " + mode)
    # plt.colorbar()
    # show and save the map
    plt.savefig(path + "/" + var + "_" + model + "_" + ver + "_" + scen + "_" + mode + ".png")
    plt.close()
    
def map_inlet(ax, lons, lats, data):
    # ax = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0),position=ax.get_position())
    # ax = ccrs.PlateCarree(central_longitude=180)
    # m = Basemap(projection='cyl', resolution='c', lon_0=180., ax=ax)
    # draw map features
    # draw map features
    ax.coastlines(resolution='110m')
    # m.drawcoastlines()
    # m.drawcountries()
    # Add country borders (equivalent to m.drawcountries())
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    # Fill continents with color (equivalent to m.fillcontinents(color='coral'))
    # ax.add_feature(cfeature.LAND, facecolor='coral')
    # Fill ocean/lakes with color (equivalent to lake_color='aqua')
    # ax.add_feature(cfeature.OCEAN, facecolor='aqua')
    # m.fillcontinents(color='coral', lake_color='aqua')
    # plot data on the map
    vmax = np.nanmax(np.abs(data))
    vmin = 0.
    ax.pcolormesh(lons, lats, np.squeeze(data), cmap='Reds', vmin = vmin, vmax = vmax)
    # m.pcolormesh(lons, lats, np.squeeze(data), cmap='Reds', vmin = vmin, vmax = vmax)
    # plt.colorbar(cax=ax)

def map_inlet_fld(ax, lons, lats, data):
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0),
                  position=ax.get_position())
    # draw map features
    ax.coastlines(resolution='110m')
    # Add country borders (equivalent to m.drawcountries())
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    # plot data on the map
    vmax = np.nanmax(data)
    vmin = np.nanmin(data)
    im = ax.pcolormesh(lons, lats, np.squeeze(data), cmap='RdBu_r',
                  vmin = vmin, vmax = vmax)
    plt.colorbar(im, ax=ax)

def scroll_folders(grp, pr_dir, cmip_dir):
    # logger.info('piControl parent folder: {}'.format(cmip_dir))
    logger.info('This project folder: {}'.format(pr_dir))
    for pj in project:
        pj_dir = os.path.join(pr_dir, pj)
        if os.path.isdir(pj_dir) and os.listdir(pj_dir):
            for mg in model_groups:
                mg_dir = os.path.join(pj_dir, mg)
                # logger.info('This model group folder: {}'.format(mg_dir))
                if os.path.isdir(mg_dir) and os.listdir(mg_dir):
                    logger.info('----------------------------------')
                    logger.info('MODEL GROUP: {}'.format(mg))
                    logger.info('----------------------------------')
                    models = [d for d in os.listdir(mg_dir)]
                    for m in models:
                        logger.info('----------------------------------')
                        logger.info('MODEL: {}'.format(m))
                        logger.info('----------------------------------')
                        exp_dir = os.path.join(mg_dir, m)
                        scens = os.listdir(exp_dir)
                        for ss in scenarios[pj]:
                            logger.info('----------------------------------')
                            logger.info('EXPERIMENT: {}'.format(pj))
                            logger.info('----------------------------------')
                            if ss in scens:
                                ss_dir = os.path.join(exp_dir, ss)
                                logger.info('----------------------------------')
                                logger.info('SCENARIO: {}'.format(ss))
                                logger.info('----------------------------------')
                                run = [r for r in os.listdir(ss_dir)] 
                                # logger.info('This scenario folder: {}'.format(ss_dir))
                                for rr in runs:
                                    if rr in run:
                                        logger.info('----------------------------------')
                                        logger.info('RUN: {}'.format(rr))
                                        logger.info('----------------------------------')
                                        j=0
                                        for dom in domains:
                                            logger.info('----------------------------------')
                                            logger.info('DOMAIN: {}'.format(dom))
                                            logger.info('----------------------------------')
                                            if dom in os.listdir(os.path.join(ss_dir, rr)):
                                                v_dir = os.path.join(ss_dir, rr, dom)
                                                # logger.info('This domain folder: {}'.format(v_dir))
                                                var = [v for v in os.listdir(v_dir)]
                                                for vv in vars[j][dom]:
                                                    logger.info('----------------------------------')
                                                    logger.info('VAR: {}'.format(vv))
                                                    logger.info('----------------------------------')
                                                    if vv in var:
                                                        g_dir = os.path.join(v_dir, vv)
                                                        grids = [g for g in os.listdir(g_dir)]
                                                        for gg in grids:
                                                            logger.info('----------------------------------')
                                                            logger.info('GRID: {}'.format(gg))
                                                            logger.info('----------------------------------')
                                                            ve_dir = os.path.join(g_dir,gg)
                                                            date = [da for da in os.listdir(ve_dir)]
                                                            for daa in date:
                                                                if grp == 'cnrm':
                                                                    f_dir = ve_dir
                                                                else:
                                                                    logger.info('----------------------------------')
                                                                    logger.info('VERS: {}'.format(daa))
                                                                    logger.info('----------------------------------')
                                                                    f_dir = os.path.join(ve_dir, daa)
                                                                # logger.info('This variable folder: {}'.format(f_dir))
                                                                if os.path.isdir(f_dir) and os.listdir(f_dir):
                                                                    try:
                                                                        [ofile_y, ofile_my, ofile_std] = data_crunch(
                                                                            f_dir, ss, 
                                                                            vv, bandwidth, in_year, end_year)
                                                                    except TypeError:
                                                                        continue
                                                                    cmg_dir = os.path.join(cmip_dir, mg)
                                                                    cmm_dir = os.path.join(cmg_dir, m)
                                                                    if 'esm-piControl' in os.listdir(cmm_dir):
                                                                        # logger.info("The piControl is present...")
                                                                        try:
                                                                            pi_dir = os.path.join(cmm_dir, 'esm-piControl', rr)
                                                                            os.listdir(pi_dir)
                                                                        except FileNotFoundError:
                                                                            pi_dir = os.path.join(cmm_dir, 'esm-piControl', 'r1i1p1f1')
                                                                        try:
                                                                            hi_dir = os.path.join(cmm_dir, 'esm-hist', rr)
                                                                            os.listdir(hi_dir)
                                                                        except FileNotFoundError:
                                                                            hi_dir = os.path.join(cmm_dir, 'esm-hist', 'r1i1p1f1')
                                                                        piv_dir = os.path.join(pi_dir, dom) 
                                                                        if not os.path.isdir(piv_dir):
                                                                            if dom=='Amon':
                                                                                piv_dir = os.path.join(pi_dir, 'APmon')
                                                                            elif dom=='Lmon':
                                                                                piv_dir = os.path.join(pi_dir, 'LPmon')
                                                                            elif dom=='Omon':
                                                                                piv_dir = os.path.join(pi_dir, 'OPmon')
                                                                        pivv_dir = os.path.join(piv_dir, vv, gg)
                                                                        if grp == 'cnrm':
                                                                            fpi_dir = pivv_dir
                                                                        else:
                                                                            verspi = [vep for vep in os.listdir(pivv_dir)]
                                                                            fpi_dir = os.path.join(pivv_dir, verspi[0])
                                                                        hiv_dir = os.path.join(hi_dir, dom)
                                                                        if not os.path.isdir(hiv_dir):
                                                                            if dom=='Amon':
                                                                                hiv_dir = os.path.join(hi_dir, 'APmon')
                                                                                dom2='APmon'
                                                                            elif dom=='Lmon':
                                                                                hiv_dir = os.path.join(hi_dir, 'LPmon')
                                                                                dom2='LPmon'
                                                                            elif dom=='Omon':
                                                                                hiv_dir = os.path.join(hi_dir, 'OPmon')
                                                                                dom2='OPmon'
                                                                            else:
                                                                                dom2=dom
                                                                        hivv_dir = os.path.join(hiv_dir, vv, gg)
                                                                        if grp == 'cnrm':
                                                                            fhi_dir = hivv_dir
                                                                        else:
                                                                            vershi = [veh for veh in os.listdir(hivv_dir)]
                                                                            fhi_dir = os.path.join(hivv_dir, vershi[0])
                                                                        logger.info('This piControl folder: {}'.format(fpi_dir))
                                                                        logger.info('This historical folder: {}'.format(fhi_dir))
                                                                        if os.path.isdir(fpi_dir) and os.listdir(fpi_dir) and os.path.isdir(fhi_dir) and os.listdir(fhi_dir):   
                                                                            try:
                                                                                logger.info('Now crunching piControl data (good luck!)...')
                                                                                [ofile_piy, ofile_pimy, ofile_pistd] = data_crunch(
                                                                                    fpi_dir, 'esm-piControl', vv, 
                                                                                    bandwidth, in_year, end_year)
                                                                                logger.info('piControl data crunched!')
                                                                            except TypeError:
                                                                                continue
                                                                            #Computing tipping indicators
                                                                            try:
                                                                                [time, lon, 
                                                                                lat, unit, data, 
                                                                                indicators, masks] = tips(ofile_my, ofile_std, 
                                                                                                            ofile_pimy, ofile_pistd, 
                                                                                                            vv, yrmaxchange)
                                                                            except TypeError as e:
                                                                                    continue
                                                                            #Mapping masks and indicators
                                                                            try:
                                                                                [ofile_hiy, ofile_himy, ofile_histd] = data_crunch(
                                                                                    fhi_dir, 'esm-hist', vv, 
                                                                                    bandwidth, 1850, 2014)
                                                                            except TypeError:
                                                                                continue
                                                                            lonm = np.array(lon)
                                                                            latm = np.array(lat)
                                                                            path_f = '{}/{}/{}/{}/{}/{}'.format(path_l, grp, m, ss, rr, vv) 
                                                                            try:
                                                                                os.makedirs(path_f)
                                                                            except OSError:
                                                                                pass
                                                                            map(path_f, lon, lat, 
                                                                                indicators[0], vv, daa, m, 
                                                                                ss, 'std')
                                                                            map(path_f, lon, lat, 
                                                                                indicators[1], vv, daa, m, 
                                                                                ss, 'maxch')
                                                                            map(path_f, lon, lat, 
                                                                                indicators[2], vv, daa, m, 
                                                                                ss, 'sym')
                                                                            map(path_f, lon, lat, 
                                                                                indicators[1], vv, daa, m, 
                                                                                ss, 'mmod')
                                                                            map(path_f, lon, lat, 
                                                                                masks[0], vv, daa, m, 
                                                                                ss, 'mask_std')
                                                                            map(path_f, lon, lat, 
                                                                                masks[1], vv, daa, m, 
                                                                                ss, 'mask_maxch')
                                                                            map(path_f, lon, lat, 
                                                                                masks[2], vv, daa, m, 
                                                                                ss, 'mask_pc99')
                                                                            map(path_f, lon, lat,
                                                                                masks[3], vv, daa, m, 
                                                                                ss, 'mask_combine')
                                                                            map(path_f, lon, lat,
                                                                                masks[4], vv, daa, m, 
                                                                                ss, 'mask_sym')
                                                                            map(path_f, lon, lat,
                                                                                masks[5], vv, daa, m, 
                                                                                ss, 'mask_mmod')
                                                                            clusters, masks_c = oet.analyze.clustering.build_cluster_mask(
                                                                                            np.array(np.bool_(np.squeeze(masks[3]))),
                                                                                            latm,
                                                                                            lonm,
                                                                                            max_distance_km='infer',
                                                                                            min_samples=8)
                                                                            if len(clusters) >= 1:
                                                                                clusters = np.array(clusters, dtype=int)
                                                                                logger.info('Cluster size: {}'.format(np.shape(clusters)))
                                                                                plotting_clusters(
                                                                                    path_f, ofile_y, ofile_piy, ofile_hiy,
                                                                                    ofile_my, ofile_pimy, ofile_himy,
                                                                                    clusters, time, lon, lat, unit, data, 
                                                                                    indicators, vv, daa, m, 'combine', ss,
                                                                                    thres_gp)
                                                                            os.remove(ofile_piy)
                                                                            os.remove(ofile_pistd)
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
                                logger.info("There's nothing in scenario {}.".format(ss))
                        else:
                            logger.info("Experiment {} is not available".format(ss))
                else:
                    logger.info("No model is found in {} model group".format(mg))
        else:
            logger.info("Project {} is empty".format(pj))
            
mipp=0

logger.info('This is the OPTIMESM algorithm for detection of moderate non-linear surprises, vers. {}'.format(hunt_vers))
logger.info('Created by Valerio Lembo (CNR-ISAC).')
logger.info('Algorithm parameters:')
logger.info('Moving average window (in years): {}'.format(bandwidth))
logger.info('Maximum jump over: {} years'.format(yrmaxchange))
logger.info('Minimum length of the datasets (in yeasrs): {}'.format(minnofyears))
logger.info('Percentile thresholds for last 30 years of scenario runs: {} and {}'.format(pc[y30_pcmin],pc[y30_pcmax]))
logger.info('Threshold for maximum jump in percentile: {}'.format(pc[maxj_pcmax]))
logger.info('Threshold for standard deviation in percentile: {}'.format(pc[std_pcmax]))
logger.info('Threshold for Kolmogorov t-Student test for normality: {}'.format(pc[oett_pcmax]))
logger.info('Threshold for Dip test for bimodality: {}'.format(pc[oetd_pcmax]))
logger.info('Minimal number of gridpoints for clustering: {}'.format(thres_gp))
logger.info('----------------------------------------------')

logger.info('Starting the loops on the files...')
for grp in folders:
    if grp == 'ipsl':
        grp_dir = '/g100_store/DRES_OptimESM/ESGF/external/20250619'
    else:
        grp_dir = os.path.join(path, grp)
    logger.info('----------------------------------')
    logger.info('OPTIMESM folder: {}'.format(grp_dir))
    logger.info('----------------------------------')
    if os.path.isdir(grp_dir) and os.listdir(grp_dir):
        if grp == 'smhi':
            pr_dir = os.path.join(grp_dir, 'CMIP6Plus')
            cmip_dir = os.path.join(path, 'smhi/CMIP6Plus/CMIP')
            scroll_folders(grp, pr_dir, cmip_dir)
        elif grp == 'ipsl':
            pr_dir = os.path.join(grp_dir, 'CMIP6Plus')
            cmip_dir = '/g100_store/DRES_OptimESM/ESGF/external/20250619/CMIP6Plus/CMIP'
            scroll_folders(grp, pr_dir, cmip_dir)
        else:
            vers = [ve for ve in os.listdir(grp_dir)]
            for vee in vers:
                logger.info('DATE: {}'.format(vee))
                aux = os.path.join(grp_dir, vee)
                if grp=='mohc':
                    pr_dir = os.path.join(aux, 'CMIP6')
                    cmip_dir = os.path.join(path, 'mohc/20250109/CMIP6/CMIP')
                elif grp == 'cnrm':
                    pr_dir = os.path.join(aux, 'CMIP6plus')
                    cmip_dir = os.path.join(path, 'cnrm/20240805/CMIP6/CMIP')
                else:
                    if grp=='bsc':
                        pr_dir = os.path.join(aux, 'CMIP6')
                    elif grp =='knmi':
                        pr_dir = os.path.join(aux, 'gw23/CMIP6Plus')
                    else:
                        pr_dir = os.path.join(grp_dir, 'CMIP6Plus')
                    cmip_dir = os.path.join(path, 'smhi/CMIP6Plus/CMIP')
                scroll_folders(grp, pr_dir, cmip_dir)
    else:
        logger.info("This OPTIMESM folder is empty...")
mipp = mipp + 1
logger.info("Finished hunting for tipping. Now rest...")
