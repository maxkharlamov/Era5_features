# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:25:07 2020

@author: kharl
"""

import xarray as xr
import pandas as pd
import numpy as np

import geopandas as gpd

import os
from tqdm import tqdm

import matplotlib.pyplot as plt

def sel_mon(month):
    return (month >= 3) & (month <= 5)

def clip_by_shape(xr_file, shape):           #return nc mask
    ddd = xr_file[list(xr_file.variables)[3]].copy()
    ddd = ddd.mean(dim = 'time')
    
    # перевод netcdf в geodataframe
    df = ddd.to_dataframe()
    df = df.reset_index()
    geom=gpd.points_from_xy(df['longitude'], df['latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    
    within = []
    for i in range(len(gdf)):
        aaa = gdf['geometry'].loc[i].within(shape['geometry'].loc[0])
        if aaa == False:
            aaa = np.nan
        else:
            aaa = 1
        within.append(aaa)
        
    gdf['within'] = within
    
    gdf = gdf.set_index(['latitude', 'longitude'])
    nc_mask = gdf.to_xarray()
    #xr_masked = xr_file * nc_mask['within']
    
    return nc_mask['within']


def zonalmean_xr(xr_dataset_mask, shape):
    vars_ = list(xr_dataset_mask.variables)
    vars_ = vars_[3:]
    df = pd.DataFrame()
    
    for v in vars_:
        df[v] = xr_dataset_mask[v].mean(dim = ['latitude', 'longitude']).to_pandas()
    return df

# =============================================================================
# datasets
# =============================================================================
#print('reading datasets...')
plt.ioff()
directory_to_save_graphs = 'reanalysis_output/clip_by_shape/level1/graphs/'
directory_to_save = 'reanalysis_output/clip_by_shape/level1/'

SD = xr.open_dataset('reanalysis_output/level1_ERA5/SD_full.nc')
temp_prec = xr.open_dataset('reanalysis_output/level1_ERA5/Temp_Prec_SD_full4.nc')
fr_depth = xr.open_dataset('reanalysis_output/level1_ERA5/fr_depth_stat_full.nc')
SM = xr.open_dataset('reanalysis_output/level1_ERA5/SM_stat_full2.nc')      # SM_stat_full2 - без ошибок

#SM_lin = xr.open_dataset('reanalysis_output/level1_ERA5/SM/SM_lin_full.nc')
#SM_type = xr.open_dataset('reanalysis_output/level1_ERA5/SM/SM_type_full.nc')


# =============================================================================
# shapes
# =============================================================================
#print('reading shapes...')
errors = []
folders = os.listdir('shapes/shapes_etr/')
for folder in folders:
    if not os.path.exists(directory_to_save + folder + os.sep + 'graphs/'): os.makedirs(directory_to_save + folder + os.sep + 'graphs/')
    
    shapes_path = 'shapes/shapes_etr/' + folder
    shapes = os.listdir(shapes_path)
    shapes = [x for x in shapes if x.endswith('.shp')]
    
    # =============================================================================
    # zonal_mean
    # =============================================================================
    #print('zonal mean...')
    shape = gpd.read_file(shapes_path + os.sep + shapes[0])
    for station in tqdm(shapes[:], desc = 'zonal mean...'):
        try:
            shape = gpd.read_file(shapes_path + os.sep + station)
            shape = shape.to_crs(epsg=4326)
            
            mask = clip_by_shape(SD, shape)
            
            #zonal_mean
            SD_df = zonalmean_xr(SD*mask, shape = shape)
            temp_prec_df = zonalmean_xr(temp_prec*mask, shape = shape)
            fr_depth_df = zonalmean_xr(fr_depth*mask, shape = shape)
            SM_df = zonalmean_xr(SM*mask, shape = shape)
            
            df = pd.concat([SD_df, temp_prec_df, SM_df, fr_depth_df], axis = 1)

            df = df.replace([np.inf, -np.inf], 0)
            df.to_csv(directory_to_save + folder + os.sep + station[:-4] + '.csv')
        except:
            errors.append(station)

