"""
Spyder Editor

This is a temporary script file.
"""

from cdo import Cdo
from netCDF4 import Dataset as ds
import os
import glob
import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

cdo = Cdo()

def map(path, lons, lats, data, var, model, scen, mode, rea):
    m = Basemap(projection='cyl', resolution='c', lon_0=180.)
    lons[lons > 180.] -= 360.
    lons_2 = lons[lons>=0]
    lons_3 = np.append(lons[lons<0], lons_2, 0)
    lons = lons_3 + 180.
    # read data from netCDF file
    # draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='coral', lake_color='aqua')
    # plot data on the map
    # m.readshapefile("path/to/shapefile", "shapes")
    vmax = np.nanmax(np.abs(data))
    vmin = -vmax
    m.pcolormesh(lons, lats, np.squeeze(data), cmap='seismic', vmin = vmin, vmax = vmax)
    # add title
    plt.title(var + " " + model + " " + scen + " " + mode + " " + rea)
    plt.colorbar()
    # show and save the map
    plt.show()
    plt.savefig(path + "/" + var + "_" + model + "_" + scen + "_" + rea + "_" + mode + "_diff.png")
    plt.close()

def tips(filein,filein_std,filein_pi,filein_pistd,varname,yrmxch):            
    # file_amoc = dir+'/{}_AMOC_idx.nc'.format(m)
    # cdo.mergetime('lat,{}'.format(lat_model), input=files, output='aux.nc')
    data = ds(filein)
    var = data.variables[varname][:,:,:]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    time = data.variables['time'][:]
    nyrs = len(time)
    usedyears = 2 * yravg
    if nyrs <= usedyears:
        print('Not enough years')
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
    
    var_shift = np.zeros(np.shape(var))
    var_ini = var[:yrmxch,:,:] 
    var_end = var[fin:,:,:] 
    var_shift[yrmxch:,:,:] = var[:fin,:,:]
    var_shiftdiff = var[yrmxch:,:,:]-var_shift[yrmxch:,:,:]
    var_diffabs = np.squeeze(np.abs(var_shiftdiff))
    var_timmax = np.nanmax(var_diffabs,0)
    
    varpi_shift = np.zeros(np.shape(varpi))
    varpi_shift[yrmxch:,:,:] = varpi[:finpi,:,:]
    varpi_shiftdiff = varpi[yrmxch:,:,:]-varpi_shift[yrmxch:,:,:]
    varpi_diffabs = np.squeeze(np.abs(varpi_shiftdiff))
    varpi_timmax = np.nanmax(varpi_diffabs,0)

    vardiff_timmax = var_timmax - varpi_timmax
    vardiff_std = var_std - varpi_std
    
    var_ini_tm = np.nanmean(var_ini, 1)
    var_end_tm = np.nanmean(var_end, 1)
    var_diff = np.squeeze(var_end_tm - var_ini_tm)

    indicators = [vardiff_std, vardiff_timmax, var_diff]
    # print(np.shape(vardiff_timmax))
    # var1[1:dims[0]] = bn.move_mean(var1[1:dims[0]], window=gwindow,
    #                                   min_count=1)
    return lon, lat, indicators


bandwidth = 10
yravg = 5
yrmaxchange = 10
minnofyears = 11

model_groups = [
    'AS-RCEC', 'AWI', 'BCC', 'CAMS', 'CAS', 'CCCma', 'CCCR-IITM', 'CMCC', 
    'CSIRO', 'CSIRO-ARCCSS', 'E3SM-Project', 'EC-Earth-Consortium', 'FIO-QLNM',
    'HAMMOZ-Consortium', 'INM', 'IPSL', 'KIOST', 'MIROC', 'MOHC', 'MPI-M',
    'MRI', 'NASA-GISS', 'NCAR', 'NCC', 'NIMS-KMA', 'NOAA-GFDL', 'NUIST', 'SNU',
    'THU']
#, 'UA']
vars = [
        {'Amon':['tas']},
        {'SImon':['']}]

scenarios = ['abrupt-4xCO2']

runs = ['r1i1p1f1', 'r2i1p1f1']

path = '/work_big/datasets/synda/data/CMIP6/CMIP'
path_l = '/home/lembo/tipping_optimesm'

for mg in model_groups:
    mg_dir = os.path.join(path, mg)
    if os.listdir(mg_dir):
        models = [d for d in os.listdir(mg_dir)]
        print(models)
        for m in models:
            m_dir = os.path.join(mg_dir, m)
            scen = [s for s in os.listdir(m_dir)]
            print(scen)
            if os.listdir(m_dir):
                for ss in scenarios:
                    if (ss in scen):
                        print(ss)
                        r_dir = os.path.join(m_dir, ss)
                        run = [r for r in os.listdir(r_dir)]
                        for rr in runs:
                            if rr in run:
                                print(rr)
                                v_dir = os.path.join(r_dir, rr, 'Amon')
                                var = [v for v in os.listdir(v_dir)]
                                for vv in vars[0]['Amon']:
                                    print(vv)
                                    if vv in var:
                                       g_dir = os.path.join(v_dir, vv)
                                       grids = [g for g in os.listdir(g_dir)]
                                       for gg in grids:
                                           ve_dir = os.path.join(g_dir,gg)
                                           vers = [ve for ve in os.listdir(ve_dir)]
                                           for vee in vers:
                                                f_dir = os.path.join(ve_dir, vee)
                                                files = [f for f in os.listdir(f_dir)]
                                                os.chdir(f_dir)
                                                ofile = 'file_merged.nc'
                                                try:
                                                    os.remove(ofile)
                                                except OSError:
                                                    pass
                                                cdo.mergetime(
                                                    input=glob.glob(f_dir+'/*.nc'),
                                                    output = ofile)
                                                ofile_y = 'file_merged_y.nc'
                                                try:
                                                    os.remove(ofile_y)
                                                except OSError:
                                                    pass
                                                cdo.yearmean(
                                                    input= ofile,
                                                    output = ofile_y)
                                                ofile_rundet = os.path.join(f_dir, 'file_merged_y_rundet.nc')
                                                try:
                                                    os.remove(ofile_rundet)
                                                except OSError:
                                                    pass
                                                cdo.detrend(
                                                    input='-runmean,10 {}'.format(ofile_y),
                                                    output = ofile_rundet)
                                                ofile_std = os.path.join(f_dir, 'file_merged_std.nc')
                                                try:
                                                    os.remove(ofile_std)
                                                except OSError:
                                                    pass                                                    
                                                cdo.timstd(
                                                    input = ofile_rundet,
                                                    output = ofile_std)
                                                if 'piControl' in scen:
                                                    pi_dir = os.path.join(m_dir, 'piControl', rr, 'Amon', vv, gg)
                                                    vers = [v for v in os.listdir(pi_dir)]
                                                    print(pi_dir)
                                                    for vee in vers:
                                                        fpi_dir = os.path.join(pi_dir, vee)
                                                        files = [f for f in os.listdir(fpi_dir)]
                                                        os.chdir(fpi_dir)
                                                        ofile_pi = 'file_merged.nc'
                                                        if not ofile_pi in files:
                                                            cdo.mergetime(
                                                                input=glob.glob(fpi_dir+'/*.nc'),
                                                                output = ofile_pi)
                                                        ofile_piy = 'file_merged_y.nc'
                                                        if not ofile_piy in files:
                                                            cdo.yearmean(
                                                                input= ofile_pi,
                                                                output = ofile_piy)
                                                        ofile_pirundet = os.path.join(fpi_dir, 'file_merged_y_rundet.nc')
                                                        if not ofile_pirundet in files:
                                                            cdo.detrend(
                                                                input='-runmean,10 {}'.format(ofile_piy),
                                                                output = ofile_pirundet)
                                                        ofile_pistd = os.path.join(fpi_dir, 'file_merged_std.nc')
                                                        if not ofile_pistd in files:
                                                            cdo.timstd(
                                                                input = ofile_pirundet,
                                                                output = ofile_pistd)
                                                    [lon, lat, indicators] = tips(ofile_rundet, ofile_std, ofile_pirundet, ofile_pistd, vv, yrmaxchange)
                                                    map(path_l, lon, lat, indicators[0], vv, m, ss, 'std', 'Amon')
                                                    map(path_l, lon, lat, indicators[1], vv, m, ss, 'maxch', 'Amon')
                                                    os.remove(ofile_pi)
                                                    os.remove(ofile_piy)
                                                    os.remove(ofile_pirundet)
                                                    os.remove(ofile_pistd)
                                                else:
                                                    print('No piControl run found')
                                                os.chdir(f_dir)
                                                os.remove(ofile)
                                                os.remove(ofile_y)
                                                os.remove(ofile_rundet)
                                                os.remove(ofile_std)
                                    else:
                                        "Variable not available"
                                v_dir = os.path.join(r_dir, rr, 'SImon')
                                if os.path.isdir(v_dir):
                                    var = [v for v in os.listdir(v_dir)]
                                    for vv in vars[1]['SImon']:
                                        print(vv)
                                        if vv in var:
                                            g_dir = os.path.join(v_dir, vv)
                                            grids = [g for g in os.listdir(g_dir)]
                                            for gg in grids:
                                                ve_dir = os.path.join(g_dir,gg)
                                                vers = [ve for ve in os.listdir(ve_dir)]
                                                for vee in vers:
                                                    f_dir = os.path.join(ve_dir, vee)
                                                    files = [f for f in os.listdir(f_dir)]
                                                    print(files)
                                                    os.chdir(f_dir)
                                                    ofile = 'file_merged.nc'
                                                    if not ofile in files:
                                                        cdo.mergetime(
                                                            input=glob.glob(f_dir+'/*.nc'),
                                                            output = ofile)
                                                    ofile_y = 'file_merged_y.nc'
                                                    if not ofile_y in files:
                                                        cdo.yearmean(
                                                            input= ofile,
                                                            output = ofile_y)
                                                    ofile_rundet = 'file_merged_y_rundet.nc'
                                                    if not ofile_rundet in files:
                                                        cdo.detrend(
                                                            input='-runmean,10 {}'.format(ofile_y),
                                                            output = ofile_rundet)
                                                    ofile_std = 'file_merged_std.nc'
                                                    if not ofile_std in files:
                                                       cdo.timstd(
                                                           input = ofile_rundet,
                                                           output = ofile_std)
                                                    tippings = tips(ofile_rundet, ofile_std, ofile_pirundet, ofile_pistd, vv, yrmaxchange)
                                                    # tippings = tips(ofile_rundet,vv,yravg)
                                                    os.remove(ofile)
                                                    os.remove(ofile_y)
                                                    os.remove(ofile_rundet)
                                                    os.remove(ofile_std)
                                                    #print(files)
                                        else:
                                            "Variable not available"
                                else:
                                    "No sea-ice for this run..."
                            else:
                                "Run not available"
                    else:
                        print("Experiment not available")
            else:
                print("Directory is empty")
    else:
        print("Directory is empty")