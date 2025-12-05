#!/usr/bin/env python3

from dask.distributed import Client
import dask

def setup_dask(namelist) :

    """Function for setting up dask as a function of the 
    configuration - experimental"""

    if namelist['dask'] :
        
        client = Client(
            n_workers=namelist['workers'], 
            threads_per_worker=namelist['threads']
        )
    else : 
        dask.config.set(
            scheduler="synchronous",
            **{'array.slicing.split_large_chunks': False}
            )