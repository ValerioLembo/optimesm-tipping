#!/usr/bin/env python3

# MiLES2 functions
from string import Template
import os
import calendar

def season2timeseason(season) :
    """Convert strings of seasons to machine readable integers:
    standard and extende seasons are possible, alternatively single
    months in both short and long format can be used"""

    dseasons = { 
        'DJF': [1,2,12], 
        'DJFM': [1,2,3,12], 
        'MAM': [3,4,5], 
        'JJA': [6,7,8], 
        'JJAS': [1,2,3,12], 
        'SON': [10,11,12]
        }

    short = {month: index for index, month in enumerate(calendar.month_abbr) if month}
    long = {month: index for index, month in enumerate(calendar.month_name) if month}

    if season.upper() in dseasons.keys(): 
        out = dseasons[season.upper()]
    elif len(season) == 3 : 
        out = short[season]
    else :
        out = long[season]

    return out

def get_input_directory(data, analysis) :

    """Create the input directory using string replacement 
    combining structure from the configuration file and input 
    from the namelist"""

    project = data.get(analysis['project'])
    if isinstance(project, dict) : 
        dirdata = project[analysis['dataset']]
    else :
        dirdata = project    

    data_directory = Template(dirdata).safe_substitute(
    project = analysis['project'], 
    dataset = analysis['dataset'], 
    frequency= analysis.get('frequency', 'day'),
    variable= analysis.get('variable', 'Z500'),
    experiment = analysis.get('experiment', 'historical'),
    ensemble = analysis.get('ensemble', 'r1i1p1f1'),
    )
    
    file_path = data_directory + '/*.nc'

    return file_path

def get_output_directory(miles, analysis, create = True) :

    """Create the output directories combining info from the 
    namelist file"""
    
    basedir = miles['output']
    finaldir = os.path.join(basedir, 'Block',
        analysis['project'],
        analysis['dataset'],
        analysis.get('experiment', ''),
        analysis.get('ensemble', ''),
        str(analysis['year1']) + '_' + str(analysis['year2']),
        analysis['season']
    )
    
    if create :
        os.makedirs(finaldir, exist_ok=True) 

    return finaldir
