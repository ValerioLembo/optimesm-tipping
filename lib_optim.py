from cdo import Cdo
from math import floor
Cdo.debug = True
from netCDF4 import Dataset as ds
import diptest
import optim_esm_tools as oet
from scipy import stats
import xarray as xr
import xeofs as xe
import xesmf as xe
# import eofs
# from eofs.xarray import Eof
import esmtools
from esmtools.stats import standardize
import cftime
from cftime import num2date
# import xesmf as xe
import yaml
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from miles.diag import da12_blocking, da12_blocking_interp, sc04_blocking, tm90_blocking
from miles.performance import setup_dask
from miles.handling import get_input_directory, get_output_directory, season2timeseason
from miles.netcdf import write_output_netcdf, dataset_preprocessing
from dask.distributed import Client
import dask
import dask.array as da
from jinja2 import Template
import datetime


## User's options
# Fundamental parameters
filter_len = 10      # for moving average, in years
yravg = filter_len/2
yrmaxch = 10        # Chunk length for maximum jumps evaluation
minnofyears = 11        # Shortest dataset to be considered
in_year = 1850          # Initial year for historical/ssp simulations
end_year = 2010         # Last year for ssp simulations
pc = [1, 5, 10, 25,     # set of percentiles for tail comparisons
      75, 90, 95, 99]
thres_gp = 150           # Minimal area (in gridpoints 1x1) for cluster retrieval
plev= 50000

def data_crunch(f_dir,scen,var,filter,in_year,end_year):
    os.chdir(f_dir)
    tmp_dir = '{}tmp_{}_{}'.format(tmp_path,scen,date)
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
    if int(nyrs)<yrmaxch:
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


def data_crunch_xr(f_dir, scen, var, filter_len, in_year, end_year):
    """
    Xarray/Dask rewrite of the original CDO-based pipeline.
    Returns xarray.Dataset objects instead of file paths.
    """

    # -------------------------------------------------------
    # 1. Load and merge all files
    # -------------------------------------------------------
    files = sorted(glob.glob(os.path.join(f_dir, f"{var}_*.nc")))
    if len(files) == 0:
        raise FileNotFoundError("No input files found.")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)

    # -------------------------------------------------------
    # 2. Select years
    # -------------------------------------------------------
    ds = ds.sel(time=slice(str(in_year), str(end_year)))

    # Compute nyears
    nyears = int(len(ds.groupby("time.year")))

    # -------------------------------------------------------
    # 3. Horizontal remapping using xESMF
    # -------------------------------------------------------
    target_grid = xr.Dataset({
        "lat": ("lat", np.linspace(-89.5, 89.5, 180)),
        "lon": ("lon", np.linspace(0.5, 359.5, 360)),
    })

    regridder = xe.Regridder(ds, target_grid, "bilinear", reuse_weights=False)
    ds_regrid = regridder(ds[var])

    # -------------------------------------------------------
    # 4. Yearmean
    # -------------------------------------------------------
    ds_y = ds_regrid.groupby("time.year").mean("time")

    # -------------------------------------------------------
    # 5. Apply runmean and missing -> 0 for specific variables
    # -------------------------------------------------------
    def runmean(da, N):
        return da.rolling(year=N, center=True).mean()

    if scen in ["ssp585", "ssp245", "ssp126"]:
        if var in ["siconc", "sos", "tos"]:
            ds_my = runmean(ds_y, filter_len).fillna(0)
        elif var in ["zg", "ta", "ua", "va"]:
            if plev is None:
                raise ValueError("plev required for upper-air variables")
            ds_y = ds_y.sel(level=plev)
            ds_my = runmean(ds_y, filter_len)
        else:
            ds_my = runmean(ds_y, filter_len)
    else:
        if var in ["siconc", "sos", "tos"]:
            ds_my = runmean(ds_y, filter_len).fillna(0)
        elif var in ["zg", "ta", "ua", "va"]:
            if plev is None:
                raise ValueError("plev required for upper-air variables")
            ds_y = ds_y.sel(level=plev)
            ds_my = runmean(ds_y, filter_len)
        else:
            ds_my = runmean(ds_y, filter_len)

    # -------------------------------------------------------
    # 6. Standard deviation of detrended series
    # -------------------------------------------------------
    ds_dt = xr.apply_ufunc(
        lambda x: x - np.polyval(np.polyfit(np.arange(x.size), x, 1), np.arange(x.size)),
        ds_my,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds_my.dtype],
    )

    ds_std = ds_dt.std("year")

    return ds_y.to_dataset(name=var), ds_my.to_dataset(name=f"{var}_runmean"), ds_std.to_dataset(name=f"{var}_std")


def julian_date_to_decimal_years(jd):
    # Calculate Julian centuries (T) since 1850-01-01
    ref_date = datetime.datetime(1850, 1, 1)
    target_date = ref_date + np.vectorize(lambda d: datetime.timedelta(days=int(d)))(jd)
    decimal_years = []
    for date in target_date:
        decimal_year = date.year + (date.timetuple().tm_yday -1)/365.25
        decimal_years.append(decimal_year)
    decimal_years = np.array(decimal_years)
    return decimal_years


def psl_index(logger, namelist_f, path):
    logger.info("Updating namelist...")
    with open(namelist_f, 'r') as stream:
        namelist = yaml.safe_load(stream)
    #Read the config template as a string
    with open("config_miles_tmpl.yml", "r") as stream:
        template_str = stream.read()
    #Replace the placeholders in the template with values read from the namelist
    template = Template(template_str)
    filled_yaml = template.render(namelist)
    #Save the new config file
    data = yaml.safe_load(filled_yaml)
    config = data
    with open("config_miles.yml", "w") as stream:
        yaml.dump(data, stream, default_flow_style=False)
    file_dir = config['data']['CMIP6']
    file_path = file_dir + '/*.nc'
    logger.info('Reading files stored in: %s', file_path)
    logger.info('Loading the dataset and preprocessing it')
    psl = xr.open_mfdataset(file_path, chunks={'time': 100}, engine='netcdf4', use_cftime=True)
    analysis = namelist['analysis']
    # load_season = season2timeseason(analysis['season'])
    # season selection
    # year selection
    out = psl.sel(time=slice(str(analysis['year1']),str(analysis['year2'])),
                  lon=slice(0, 360), lat=slice(20, 80))

    out_mon = out.resample(time='1M').mean()
    logger.info(out_mon)
    gb = out_mon.groupby(out_mon['time'].dt.month)
    gb_mean = out_mon.groupby(out_mon['time'].dt.month)
    out_mon = gb - gb_mean.mean(dim='time')
    logger.info(out_mon)
    # out = out.sel(time=psl.time.dt.month.isin(load_season))
    
    # weights = np.sqrt(np.cos(np.deg2rad(out.lat)))
    # weights.name = "weights"
    # weights_3d = weights.broadcast_like(out).astype('float32')
    # # logger.info(out)
    # out_weighted = out
    # out_weighted['psl'] = out_weighted['psl'] * weights_3d
    # out_weighted.attrs = out.attrs
    # logger.info(out_weighted)
    # out_weighted.name = out.name
    # dayofyear = xr.DataArray(
    #     [t.timetuple().tm_yday for t in out_weighted.time.values],
    #     coords=[out_weighted.time],
    #     dims="time"
    # )
    # daily_clim = out_weighted.groupby(dayofyear).mean('time')
    # logger.info(daily_clim)
    # out_anom = out_weighted.groupby(dayofyear) - daily_clim
    # out_anom.attrs = out_weighted.attrs
    # out_anom.name = out_weighted.name + "_anom"
    # times_cftime = out_anom.time.values
    # solver = Eof(out_weighted['psl'])
    model = xe.single.EOF(n_modes=10, 
                          standardize=True, 
                          use_coslat=True,
                          random_state=None, 
                          solver='auto')
    model.fit(out_mon['psl'], dim="time")
    rotator = xe.single.EOFRotator(n_modes=10, 
                                   power=1,
                                   max_iter=None, 
                                   rtol=1e-8)
    rotator.fit(model)
    pcs = model.scores(normalized=True)
    nao_stand = pcs[0]
    pna_stand = pcs[1]
    
    times_cftime = out_mon.time.values
    # Convert each cftime object to a Python datetime.datetime
    times_py = np.array([
        datetime.datetime(
            t.year, t.month, t.day,
            t.hour, t.minute, t.second,
            t.microsecond
        ) for t in times_cftime
    ])
    
    nao_stand = nao_stand.assign_coords(time=times_py.astype('datetime64[ns]'))
    nao_stand = nao_stand.rolling(time=10*12, center=True).mean()
    pna_stand = pna_stand.assign_coords(time=times_py.astype('datetime64[ns]'))
    pna_stand = pna_stand.rolling(time=10*12, center=True).mean()
    print(nao_stand.min())
    print(nao_stand.max())
    plt.plot(nao_stand,
             color='red',
             linestyle='-',
             linewidth=2)
    plt.xlabel("Years")
    plt.ylabel("NAO stand. index; 10yr running mean")
    plt.title(analysis['dataset'])
    plt.grid(True)
    fname = 'NAO_tser_{}_{}.png'.format(analysis['dataset'], analysis['experiment'])
    fpath = path / analysis['dataset'] / fname
    plt.savefig(fpath)
    plt.close()
    print(pna_stand.min())
    print(pna_stand.max())
    plt.plot(pna_stand,
             color='blue',
             linestyle='-',
             linewidth=2)
    plt.xlabel("Years")
    plt.ylabel("PNA stand. index; 10yr running mean")
    plt.title(analysis['dataset'])
    plt.grid(True)
    fname = 'PNA_tser_{}_{}.png'.format(analysis['dataset'], analysis['experiment'])
    fpath = path / analysis['dataset'] / fname
    plt.savefig(fpath)
    plt.close()
    return [nao_stand, pna_stand]

    
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


def tel_indices(logger, namelist_f, path):
    #Loading the namelist file
    logger.info("Updating MiLES namelist...")
    with open(namelist_f, 'r') as stream:
        namelist = yaml.safe_load(stream)
    #Read the config template as a string
    with open("config_miles_tmpl.yml", "r") as stream:
        template_str = stream.read()
    #Replace the placeholders in the template with values read from the namelist
    template = Template(template_str)
    filled_yaml = template.render(namelist)
    #Save the new config file
    data = yaml.safe_load(filled_yaml)
    config = data
    with open("config_miles.yml", "w") as stream:
        yaml.dump(data, stream, default_flow_style=False)
    file_dir = config['data']['CMIP6']
    file_path = file_dir + '/*.nc'
    # print(file_path)
    #Load the dataset with xarray, preprocess it slicing the relevant season according to the namelist
    logger.info('Loading the dataset and preprocessing it')
    zg = xr.open_mfdataset(file_path, chunks={'time': 100}, engine='netcdf4', use_cftime=True)
    analysis = namelist['analysis']
    zr = dataset_preprocessing(zg, analysis)
    updated = zr
    lat = updated["lat"].data
    lon = updated["lon"].data
    lstep = abs(float((lat[1]-lat[0])/2.5))
    # logger.info('What is in updated: %s:', updated)
    if lstep.is_integer():
        logger.info('Computing the blocking')
        updated = tm90_blocking(updated)
        updated = da12_blocking(updated)
        updated = sc04_blocking(updated)
        updated_out = updated
    else:
        logger.info('Interpolating to a reasonable grid')
        lat_new = xr.Dataset({"lat": (["lat"], np.arange(90, 0, -1.5), {"units": "degrees_north"}),
                              "lon": (["lon"], lon, {"units": "degrees_east"}),
                             })
        grid_out = xe.Regridder(updated, lat_new, "conservative")
        updated_out = grid_out(updated, keep_attrs=True)
        logger.info('Computing the blocking')
        updated_out = tm90_blocking(updated_out)
        updated_out = da12_blocking(updated_out)
        updated_out = sc04_blocking(updated_out)
    updated_out.compute()
    # logger.info('What is in updated_out: %s:', updated_out)
    # print(updated_out)
    avg = updated_out.mean(dim='time').compute() * 100
    lon = updated_out.lon.values
    ver = namelist['analysis']['version']
    model = namelist['analysis']['dataset']
    scen = namelist['analysis']['experiment']
    tm90 = avg.tm90.compute().values
    plot_lonsec(path, lon, tm90, 'blocking_tm90', ver, model, scen)
    da98 = avg.da98.compute().values
    plot_lonsec(path, lon, da98, 'blocking_da98', ver, model, scen)
    # logger.info('Blocking min and max: %s. %s', updated_out.tm90.min().compute().values, updated_out.tm90.max().compute().values) 
    # for field in ['tm90', 'da98'] :
    #     avg[field].plot()
    updated_eur = updated_out.sel(lon=slice(-27, 48)).compute() #From Davini and D'Andrea 2016
    updated_pac = updated_out.sel(lon=slice(116, 180)).compute()
    amean_eur = updated_eur.mean(dim='lon') * 100
    amean_pac = updated_pac.mean(dim='lon') * 100
    # logger.info('Pac-blocking min and max: %s, %s', updated_pac.tm90.min().compute().values, updated_pac.tm90.max().compute().values)
    # logger.info('Eur-blocking min and max: %s, %s', updated_eur.tm90.min().compute().values, updated_eur.tm90.max().compute().values)
    amean_eur = amean_eur.groupby('time.year').mean(dim='time')
    amean_pac = amean_pac.groupby('time.year').mean(dim='time')
    amean_eur = amean_eur.rolling(year=10, center=True).mean()
    amean_pac = amean_pac.rolling(year=10, center=True).mean()
    amean_eur.compute()
    amean_pac.compute()
    return [amean_eur, amean_pac]


def tips_xr(ds, ds_std, ds_pi, ds_pistd, varname):
    # ------------------------------------------------------------------
    # 1. Extract fields
    # ------------------------------------------------------------------
    da      = ds[varname]           # (year, lat, lon)
    da_std  = ds_std[varname]
    da_pi   = ds_pi[varname]
    da_pistd = ds_pistd[varname]

    lat = da.lat.values
    lon = da.lon.values
    years = da.year.values
    nyrs = len(years)
    usedyears = 2 * yravg

    if nyrs <= usedyears:
        print("Not enough years")
        return
    
    fin = nyrs - yrmaxch

    # ------------------------------------------------------------------
    # 2. PI reference statistics
    # ------------------------------------------------------------------
    nyrs_pi = len(da_pi.year)
    finpi = nyrs_pi - yrmaxch

    pcm_std = np.percentile(da_pistd.values, pc, axis=0)   # (pct, lat, lon)

    # Shifted differences – xarray version
    da_shift  = xr.zeros_like(da)
    da_shift[yrmaxch:,:,:] = da[:fin,:,:]
    da_shiftdiff = da[yrmaxch:,:,:] - da_shift[yrmaxch:,:,:]
    da_timmax = np.abs(da_shiftdiff).max("year")

    da_pi_shift  = xr.zeros_like(da_pi)
    da_pi_shift[yrmaxch:,:,:] = da_pi[:finpi,:,:]
    da_pi_shiftdiff = da_pi[yrmaxch:,:,:] - da_pi_shift[yrmaxch:,:,:]
    da_pi_timmax = np.abs(da_pi_shiftdiff).max("year")

    pcm_jump = np.percentile(da_pi_timmax.values, pc, axis=0)
    pcm = np.percentile(da_pi.values, pc, axis=0)
   
    oett = np.zeros(np.shape(var_timmax))
    oetd = np.zeros(np.shape(var_timmax))
    varn = da.where(da != 0).values  # NaN where zero
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
    varpi_shift[yrmaxch:,:,:] = varpi[:finpi,:,:]
    varpi_shiftdiff = varpi[yrmaxch:,:,:]-varpi_shift[yrmaxch:,:,:]
    varpi_diffabs = np.squeeze(np.abs(varpi_shiftdiff))
    varpi_timmax = np.nanmax(varpi_diffabs,0)
    pcm_jump = np.percentile(varpi_timmax, pc)
    
    var_ini = var[:yrmaxch,:,:] 
    var_end = var[fin:,:,:] 
    var_ini_tm = np.nanmean(var_ini, 0)
    var_end_tm = np.nanmean(var_end, 0)
    
    pcm = np.percentile(varpi, pc, axis=0)
    
    mask_std = ((np.abs(da_std) > pcm_std[7,:,:])
                & (np.abs(da_pistd) != 0)
                & (~np.isnan(da_pistd))).astype(int)

    mask_max = ((np.abs(da_timmax) > pcm_jump[7,:,:])
                & (np.abs(da_pi_timmax) != 0)
                & (~np.isnan(da_pi_timmax))).astype(int)

    mask_99p = (
        ((da > pcm[7,:,:]) | (da < pcm[3,:,:])) &
        (da != 0) & (~np.isnan(da))
    ).astype(int)

    mask_99ps = (mask_99p[-30:,:,:].sum("year") / 30 == 1).astype(int)

    mask_ps = (
        (oett > (pc[2]/100)) &
        (oett != 0) &
        (~np.isnan(oett))
    ).astype(int)

    mask_pd = (
        (oetd < (pc[2]/100)) &
        (oetd != 0) &
        (~np.isnan(oetd))
    ).astype(int)

    mask_combine = (
        mask_std +
        mask_max +
        mask_ps +
        (mask_99p[-30:,:,:].sum("year") / 30)
    ) >= 3
    mask_combine = mask_combine.astype(int)
    mask_combine = np.where(
        ((mask_std.astype(int) + 
          mask_max.astype(int) +
          mask_ps.astype(int) +
          np.nansum(mask_99p[-30:,:,:].astype(int),axis=0)/30)>=3).astype(bool),
        1, 0)
    
    # ------------------------------------------------------------------
    # 5. Return results as xarray containers
    # ------------------------------------------------------------------
    indicators = xr.Dataset({
        "std_indicator": da_std,
        "timmax_indicator": da_timmax,
        "ks_pvalue": (("lat","lon"), oett),
        "dip_pvalue": (("lat","lon"), oetd)
    })

    masks = xr.Dataset({
        "mask_std": mask_std,
        "mask_max": mask_max,
        "mask_99ps": mask_99ps,
        "mask_combine": mask_combine,
        "mask_ps": (("lat","lon"), mask_ps),
        "mask_pd": (("lat","lon"), mask_pd),
    })

    return years, lon, lat, da.attrs.get("units",""), da, indicators, masks

    
def tips(logger, block, block_pi, varname, yrmaxch, pc):
    tm90 = [ds.tm90.compute().values for ds in block]
    # tm90 = block.tm90.compute().values
    # da98 = block.da98.compute().values
    tm90_eur = tm90[0]
    tm90_pac = tm90[1]
    nyrs = len(tm90_eur)
    climlen = np.rint(nyrs/10).astype(int)
    usedyears = 2 * yravg
    if nyrs <= usedyears:
        logger.info('Not enough years')
        return
    fin = nyrs - yrmaxch
    
    tm90 = [ds.tm90.compute().values for ds in block_pi]
    # tm90 = block_pi.tm90.compute().values
    # da98 = block.da98.compute().values
    tm90_eur_pi = tm90[0]
    tm90_pac_pi = tm90[1]
    nyrs_pi = len(tm90_eur_pi)
    finpi = nyrs_pi - yrmaxch
    
    pcm = np.nanpercentile(tm90_eur_pi, pc)
    # print(pcm)
    
    varpi_std = np.nanstd(tm90_eur_pi)
    # logger.info('Standard deviation: %s', varpi_std)
    # pcm_std = np.nanpercentile(varpi_std, pc)
    # logger.info('Percentile on standard deviation: %s', pcm_std)
    
    varpi_shift = np.zeros(np.shape(tm90_eur_pi))
    varpi_shift[yrmaxch:] = tm90_eur_pi[:finpi]
    varpi_shiftdiff = tm90_eur_pi[yrmaxch:]-varpi_shift[yrmaxch:]
    varpi_diffabs = np.squeeze(np.abs(varpi_shiftdiff))
    varpi_timmax = np.nanmax(varpi_diffabs)
    pcm_jump = np.nanpercentile(varpi_timmax, pc)
    # data = ds(filein_pistd)
    # varpi_std = np.squeeze(data.variables[varname][:,:,:])
    # pcm_std = np.percentile(varpi_std, pc)

    var_eur_std = np.nanstd(tm90_eur)
    
    var_shift = np.zeros(np.shape(tm90_eur))
    var_shift[yrmaxch:] = tm90_eur[:fin]
    var_shiftdiff = tm90_eur[yrmaxch:]-var_shift[yrmaxch:]
    var_diffabs = np.squeeze(np.abs(var_shiftdiff))
    var_eur_timmax = np.nanmax(var_diffabs)
   
    varn = np.where(tm90_eur==0,np.nan,tm90_eur)
    maskn = np.where(~np.isnan(varn),1,0)
    # if np.all(maskn):
    [stat, oett_eur] = stats.kstest(varn-np.nanmean(varn), 'norm')
    [dipt, oetd_eur] = diptest.diptest(varn)

    mask = 0
    if (np.abs(var_eur_std)>varpi_std).astype(bool) & (np.abs(varpi_std)!=0.).astype(bool) & ~np.isnan(varpi_std):
        logger.info('Standard deviation for EUR blocking exceeds preindustrial variability and scores: %s', np.abs(var_eur_std))
        mask = mask+1
    else:
        logger.info('Standard deviation for EUR blocking does not exceed preindustrial variability')
    if (np.abs(var_eur_timmax)>pcm_jump[7]).astype(bool) & (np.abs(varpi_timmax)!=0.).astype(bool) & ~np.isnan(varpi_timmax):
        logger.info('Max. jump for EUR blocking exceeds preindustrial variability and scores: %s', np.abs(var_eur_timmax))
        mask = mask+1
    else:
        logger.info('Max. jump for EUR blocking does not exceed preindustrial variability')
    logger.info('Lower and upper quartiles: %s, %s', pcm[3], pcm[4])
    mask_30y = (((tm90_eur > pcm[4]) | (tm90_eur < pcm[3])) & (tm90_eur != 0.) & ~np.isnan(tm90_eur)).astype(int)
    logger.info('Blocking in the last %s years exceeding both quartiles: %s', climlen, mask_30y[-climlen:])
    logger.info('Number of years in the last %s years exceeding both quartiles: %s', climlen, np.nansum(mask_30y[-climlen:]))
    if np.nansum(mask_30y[-climlen:])/climlen==1.:
        logger.info('EUR blocking consolidates a change at the end of the run that is significant wrt. preindustrial variability')
        mask = mask+1
    else:
        logger.info('EUR blocking does not consolidate a change at the end of the run that is significant wrt. preindustrial variability')
    if (oett_eur>(pc[2]/100)).astype(bool) & (oett_eur!=0.).astype(bool) & ~np.isnan(oett_eur):
        logger.info('EUR blocking passes the t-test of hypothesis for normality')
        mask = mask+1
    else:
        logger.info('EUR blocking does not pass the t-test of hypothesis for normality')
    if (oetd_eur>(pc[2]/100)) & (oetd_eur!=0.) & ~np.isnan(oetd_eur):
        logger.info('EUR blocking passes the dip test of hypothesis for multimodality')
        mask = mask+1
    else:
        logger.info('EUR blocking does not pass the dip test of hypothesis for multimodality')

    if mask>4:
        logger.info('The time series hosts a moderate non-linear surprise candidate!')
    else:
        logger.info('The time series does not host a moderate non-linear surprise candidate!')

    pcm = np.percentile(tm90_pac_pi, pc)

    varpi_std = np.nanstd(tm90_pac_pi)
    # pcm_std = np.percentile(varpi_std, pc)
    
    varpi_shift = np.zeros(np.shape(tm90_pac_pi))
    varpi_shift[yrmaxch:] = tm90_pac_pi[:finpi]
    varpi_shiftdiff = tm90_pac_pi[yrmaxch:]-varpi_shift[yrmaxch:]
    varpi_diffabs = np.squeeze(np.abs(varpi_shiftdiff))
    varpi_timmax = np.nanmax(varpi_diffabs)
    pcm_jump = np.nanpercentile(varpi_timmax, pc)

    # data = ds(filein_pistd)
    # varpi_std = np.squeeze(data.variables[varname][:,:,:])
    # pcm_std = np.percentile(varpi_std, pc)

    var_pac_std = np.nanstd(tm90_pac)
    
    var_shift = np.zeros(np.shape(tm90_pac))
    var_shift[yrmaxch:] = tm90_pac[:fin]
    var_shiftdiff = tm90_pac[yrmaxch:]-var_shift[yrmaxch:]
    var_diffabs = np.squeeze(np.abs(var_shiftdiff))
    var_pac_timmax = np.nanmax(var_diffabs)
   
    varn = np.where(tm90_pac==0,np.nan,tm90_pac)
    maskn = np.where(~np.isnan(varn),1,0)
    # if np.all(maskn):
    [stat, oett_pac] = stats.kstest(varn-np.nanmean(varn), 'norm')
    [dipt, oetd_pac] = diptest.diptest(varn)

    mask = 0
    if (np.abs(var_pac_std)>varpi_std).astype(bool) & (np.abs(varpi_std)!=0.).astype(bool) & ~np.isnan(varpi_std):
        logger.info('Standard deviation for PAC blocking exceeds preindustrial variability and scores: %s', np.abs(var_pac_std))
        mask = mask+1
    else:
        logger.info('Standard deviation for PAC blocking does not exceed preindustrial variability')
    if (np.abs(var_pac_timmax)>pcm_jump[7]).astype(bool) & (np.abs(varpi_timmax)!=0.).astype(bool) & ~np.isnan(varpi_timmax):   
        logger.info('Max. jump for PAC blocking exceeds preindustrial variability and scores: %s', np.abs(var_pac_timmax))
        mask = mask+1
    else:
        logger.info('Max. jump for PAC blocking does not exceed preindustrial variability')
    logger.info('Lower and upper quartiles: %s, %s', pcm[3], pcm[4])
    mask_30y = (((tm90_pac > pcm[7]) | (tm90_pac < pcm[3])) & (tm90_pac != 0.) & ~np.isnan(tm90_pac)).astype(int)
    logger.info('Blocking in the last %s years exceeding both quartiles: %s', climlen, mask_30y[-climlen:])
    logger.info('Number of years in the last %s years exceeding both quartiles: %s', climlen, np.nansum(mask_30y[-climlen:]))
    if np.nansum(mask_30y[-climlen:])/climlen==1.:
        logger.info('PAC blocking consolidates a change at the end of the run that is significant wrt. preindustrial variability')
        mask = mask+1
    else:
        logger.info('PAC blocking does not consolidate a change at the end of the run that is significant wrt. preindustrial variability')
    if (oett_pac>(pc[2]/100)).astype(bool) & (oett_pac!=0.).astype(bool) & ~np.isnan(oett_pac):
        logger.info('PAC blocking passes the t-test of hypothesis for normality')
        mask = mask+1
    else:
        logger.info('PAC blocking does not pass the t-test of hypothesis for normality')
    if (oetd_pac>(pc[2]/100)) & (oetd_pac!=0.) & ~np.isnan(oetd_pac):
        logger.info('PAC blocking passes the dip test of hypothesis for multimodality')
        mask = mask+1
    else:
        logger.info('PAC blocking does not pass the dip test of hypothesis for multimodality')

    if mask>4:
        logger.info('The time series hosts a moderate non-linear surprise candidate!')
    else:
        logger.info('The time series does not host a moderate non-linear surprise candidate!')
    indicators_eur = [var_eur_std, var_eur_timmax, oett_eur, oetd_eur]
    indicators_pac = [var_pac_std, var_pac_timmax, oett_pac, oetd_pac]
    # masks = [mask_std, mask_max, mask_99ps, mask_combine, mask_ps, mask_pd]
    return indicators_eur, indicators_pac


def plotting_clusters(path, file_in, file_pin, file_hin, file_min, file_mpin, file_mhin, clusters, time, lon, lat, unit, data, ind, 
                      vv, vee, mod, method, scen, thres):
    for cl in range(len(clusters[:,0,0])):
        cl_tser = clusters[cl,:,:]
        if np.nansum(cl_tser)>thres:
            logger.info("Cluster {} of {} for criterion {} passes the threshold for the area".format(cl, 
                                                                                                     np.size(clusters,axis=0), 
                                                                                                     method))
            crate = np.nansum(np.nansum(cl_tser,axis=0))/(len(lon)*len(lat))
            
            data = ds(file_min)
            var = data.variables[vv]
            fld = cl_tser[np.newaxis,:,:] * var
            tser = np.nanmean(np.nanmean(fld,axis=2),axis=1)/crate
            data = ds(file_mhin)
            var = data.variables[vv]
            fld_hi = cl_tser[np.newaxis,:,:] * var
            tser_hi = np.nanmean(np.nanmean(fld_hi,axis=2),axis=1)/crate
            yrs = julian_date_to_decimal_years(time)
            tser_all = np.append(tser_hi, tser)
            time_all = np.linspace(1850, 1850+len(tser_all), len(tser_all))
            yrs_all = time_all.astype(int)
            # yrs_all = julian_date_to_decimal_years(time_all)
            
            std = cl_tser * ind[0]
            std_m = np.nanmean(np.where(std==0,np.nan,std))
            mjump = cl_tser * ind[1]
            mjump_m = np.nanmean(np.where(mjump==0,np.nan,mjump)) 
            # dipt = cl_tser * ind[3]
            fld_n = np.where(fld==0,np.nan,fld)
            tstd = np.nanstd(np.nanstd(fld_n,axis=2),axis=1)
            
            [dipt,pd] = diptest.diptest(tser) 
            [stat, ps] = stats.kstest(tser-np.nanmean(tser), 'norm')
            plt.figure(figsize=(12, 8))  # Set the figure size
            ax1 = plt.subplot(2,2,1)
            map_inlet(ax1, lon, lat, np.squeeze(clusters[cl,:,:]))
            ax2 = plt.subplot(2,2,2)
            plot_tser_inlet(yrs_all, tser_all, vv)
            ax3 = plt.subplot(2,2,3)
            plot_tser_inlet(yrs, tstd, 'std')
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
                                                           

def plot_lonsec(path, lon, var, vname, ver, model, scen):
    # Plotting
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(lon, var, color='blue', linewidth=1)
    plt.title(vname + " " + model + " " + ver + " " + scen)
    plt.xlabel('Longitude [deg]')  # Set the x-axis label
    plt.ylabel(vname)  # Set the y-axis label
    plt.grid(True)  # Enable gridlines
    plt.tight_layout()  # Adjust the spacing of the plot
    filename = f"{vname}_{model}_{ver}_{scen}_lonsec.png"
    filepath = path / model / scen / filename
    # make sure directories exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # print(filepath)
    # save the figure
    plt.savefig(filepath)
    plt.close()


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
    filename = f"{vname}_{model}_{ver}_{scen}_{name}_tser.png"
    filepath = path / model / scen / filename
    # make sure directories exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(filepath)
    # save the figure
    plt.savefig(filepath)
    plt.close()
    
def plot_tser_inlet(time, var, vname):
    # Plotting
    # plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(time, var, color='blue', linewidth=1)
    plt.xlabel('Years')  # Set the x-axis label
    plt.ylabel(vname)  # Set the y-axis label
    plt.grid(True)  # Enable gridlines
    # plt.tight_layout()  # Adjust the spacing of the plot

def map(path, lons, lats, data, var, ver, model, scen, mode):
    ax = ccrs.PlateCarree(central_longitude=180)
    # m = Basemap(projection='cyl', resolution='c', lon_0=180.)
    lons[lons > 180.] -= 360.
    lons_2 = lons[lons>=0]
    lons_3 = np.append(lons[lons<0], lons_2, 0)
    lons = lons_3 + 180.
    # draw map features
    ax.coastlines(resolution='110m')
    # m.drawcoastlines()
    # m.drawcountries()
    # Add country borders (equivalent to m.drawcountries())
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
    # add title
    plt.title(var + " " + model + " " + ver + " " + scen + "\n " + mode)
    plt.colorbar()
    # show and save the map
    plt.savefig(path + "/" + var + "_" + model + "_" + ver + "_" + scen + "_" + mode + ".png")
    plt.close()
    
def map_inlet(ax, lons, lats, data):
    ax = ccrs.PlateCarree(central_longitude=180)
    # m = Basemap(projection='cyl', resolution='c', lon_0=180., ax=ax)
    lons[lons > 180.] -= 360.
    lons_2 = lons[lons>=0]
    lons_3 = np.append(lons[lons<0], lons_2, 0)
    lons = lons_3 + 180.
    # draw map features
    # draw map features
    ax.coastlines(resolution='110m')
    # m.drawcoastlines()
    # m.drawcountries()
    # Add country borders (equivalent to m.drawcountries())
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
    # plt.colorbar(cax=ax)
