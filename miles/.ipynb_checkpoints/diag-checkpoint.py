#!/usr/bin/env python3

# MiLES2 functions
import xarray as xr
import numpy as np
import xesmf as xe

def da12_blocking_interp(zg, lon) :
    """Compute DA12 instantaneous blocking and return an updated 
    dataset"""

    # Latitude definition
    delta = 15
    latmax = 90 - delta
    latmin = 30

    # select geopotential for handling DataArray
    zr = zg.zg

    lat_new = xr.Dataset({
        "lat": (["lat"], np.arange(0, 90, 1.5), {"units": "degrees_north"}),
        "lon": (["lon"], lon, {"units": "degrees_east"}),
    })
    grid_out = xe.Regridder(zg, lat_new, "conservative")
    zr_out = grid_out(zr, keep_attrs=True)
    print(zr_out)
    
    # Gradients
    ghgn = (zr_out.sel(lat=slice(latmax+delta,latmin+delta)).data - zr_out.loc[:, latmax:latmin, :].data ) / delta
    ghgs = (zr_out.loc[:, latmax:latmin, :].data - zr_out.loc[:, (latmax-delta):(latmin-delta), :].data ) / delta

    # Net vector, renomarlized to the original shape
    ib = zr_out.sel(lat=slice(latmax, latmin)).rename('da12_ib')
    ib.data = ((ghgs > 0) & (ghgn < -10))
    ib = ib.reindex_like(zr_out)
    
    # merging the object
    out = xr.merge([zg, ib])

    return out

def da12_blocking(zg) :
    """Compute DA12 instantaneous blocking and return an updated 
    dataset"""

    # Latitude definition
    delta = 15
    latmax = 90 - delta
    latmin = 30

    # select geopotential for handling DataArray
    zr = zg.zg

    # Gradients
    ghgn = (zr.sel(lat=slice(latmax+delta,latmin+delta)).data - zr.loc[:, latmax:latmin, :].data ) / delta
    ghgs = (zr.loc[:, latmax:latmin, :].data - zr.loc[:, (latmax-delta):(latmin-delta), :].data ) / delta

    # Net vector, renomarlized to the original shape
    ib = zr.sel(lat=slice(latmax, latmin)).rename('da12_ib')
    ib.data = ((ghgs > 0) & (ghgn < -10))
    ib = ib.reindex_like(zr)
    
    # merging the object
    out = xr.merge([zg, ib])

    return out

def sc04_blocking(zg) :
    """"Compute the Scherrer 2004 anomaly blocking index and return 
    un updated xarray object"""

    # Latitude boundaries and quantile
    latmax = 80
    latmin = 50
    quant = 0.9

    # select geopotential for handling DataArray
    zr = zg.zg

    field = zr.sel(lat=slice(latmax,latmin))
    
    # removing daily seasonal cycle
    field = field.assign_coords(month_day=zr.time.dt.strftime("%m-%d"))
    # rechunking is necessary to compute quantiles
    anom =  field.groupby("month_day") - field.groupby("month_day").mean("time").chunk(dict(month_day=-1))

    anom_dechunk = anom.chunk(dict(time=-1))
    threshold = anom_dechunk.sel(lat=slice(latmax,latmin)).quantile(q=quant)
    sc04 = xr.where(anom>threshold, 1, 0).rename('sc04_ib').drop_vars(['month_day','quantile']).squeeze()
    sc04 = sc04.reindex_like(zr)
    
    # merging the object
    out = xr.merge([zg, sc04])

    return out


def tm90_blocking(zg) :

    delta=20
    phi_zero=60
    dlat = 5
    phi_north=phi_zero + delta 
    phi_south=phi_zero - delta

    # select geopotential for handling DataArray
    zr = zg.zg

    ghgn = (zr.sel(lat=slice(phi_north+dlat,phi_north-dlat)).data - zr.sel(lat=slice(phi_zero+dlat,phi_zero-dlat))) / delta
    ghgs = (zr.sel(lat=slice(phi_zero+dlat,phi_zero-dlat)) - zr.sel(lat=slice(phi_south+dlat,phi_south-dlat)).data) / delta

    tm90 = ((ghgs > 0) & (ghgn < -10)).any(dim='lat').rename('tm90')
    tm90 = tm90.where(True, 1, 0)

    da98 = ((ghgs > 0) & (ghgn < -5)).any(dim='lat').rename('da98')
    da98 = da98.where(True, 1, 0)

    out = xr.merge([zg, tm90, da98])

    return out



