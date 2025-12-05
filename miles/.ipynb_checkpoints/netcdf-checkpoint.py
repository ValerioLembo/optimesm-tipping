#!/usr/bin/env python3

# MiLES2 functions
from pint import UnitRegistry
from miles.handling import season2timeseason

g0 = 9.80665
ureg = UnitRegistry()

def dataset_preprocessing(xfield, namelist) :
    """Pre-process a geopotential height dataset and provide 
    the correct data-array"""

    # get the variable name:
    #for k in list(zg.data_vars): 
    #    if (len(zg.data_vars[k].shape)>2):
    #        varname = k
    for t in list(xfield.data_vars): 
        if t in ['hgt', 'var129', 'zg']: 
            varname = t

    # select seasons
    load_season = season2timeseason(namelist['season'])

    # get the vertical dimension
    for t in list(xfield.coords): 
        if t in ['level', 'plev']: 
            vertical_dim = t

    # convert vertical axis it to Pa
    zaxis = (xfield.coords[vertical_dim].data * ureg(xfield.coords[vertical_dim].units)).to('Pa')
    out = xfield.assign_coords({vertical_dim: (vertical_dim, zaxis.magnitude, {'units': 'Pa'})})
    out = out.rename({vertical_dim: 'plev'})


    # invert latitude if necessary, and subset Northern Hemisphere
    if (out.lat[-1] > out.lat[0]) : 
        out = out.reindex(lat=list(reversed(out.lat)))
    out = out.sel(lat=slice(90,0))

    # month selection and vertical selection
    out = out[varname].sel(time=xfield.time.dt.month.isin(load_season), plev = 50000)
    out= out.drop_vars('plev').squeeze()
    # year selection
    out = out.sel(time=slice(str(namelist['year1']),str(namelist['year2'])))

    # roll longitudes by 180 degrees to -180 to 180
    if max(out.lon) > 350 :
        out = out.assign_coords({"lon": (((out.lon + 180) % 360) - 180)})
        out = out.sortby(out.lon)

    # check geopotential or geopotential height
    zval = out.isel(time=0).max().values
    if (zval > 10000) :
        out = out/g0

    out = out.to_dataset(name = 'zg')

    return out


def write_output_netcdf(filename, xfield) : 

    """Set some compression options for NetCDF writing"""
    
    comp = dict(zlib=True, complevel=1)
    encoding = {var: comp for var in xfield.data_vars}
    xfield.to_netcdf(
        filename,
        format="NETCDF4", 
        encoding = encoding)