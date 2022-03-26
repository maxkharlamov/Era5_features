# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 04:02:32 2020

@author: kharl
"""
import numpy as np
import pandas as pd
import xarray as xr

from os import listdir
from itertools import product
from tqdm import tqdm

from multiprocessing import Pool
import multiprocessing as mp

def xarray_to_df(nc):
    
    xarray_list = []
    for i in tqdm(range(len(nc['longitude'][:])), desc = 'make list'):
        for j in range(len(nc['latitude'][:])):
            df_pre = nc[:, j, i].to_dataframe()
            xarray_list.append(df_pre)
            
    return xarray_list

def sd_stat(df_cut):
# =============================================================================
#     Функция для расчета характеристик снежного покрова
#     
# =============================================================================
    
    df_cut = df_cut.drop(['year'], axis = 1)
    df_stat = pd.DataFrame()
    
    df_snow = df_cut[df_cut['sd'] > 0]
    df_stat['snow_days'] = [len(df_snow)]
    
    try:
        df_stat['len_snow_period'] = [df_snow.index[-1] - df_snow.index[0]]
        df_stat['len_snow_period'] = df_stat['len_snow_period'].dt.days + 1
        df_stat['len_snowbreak'] = df_stat['len_snow_period'] - df_stat['snow_days']
    except:
        df_stat['len_snow_period'] = [0]
        df_stat['len_snowbreak'] = [np.nan]
        
    df_ind = pd.DataFrame()
    df_ind['snowdepth'] = df_snow.index
    
    aaa = pd.DataFrame()    
    aaa['snowdepth_sh'] = df_snow['sd'].shift(1).dropna().index
    
    df_ind = pd.concat([df_ind, aaa], axis = 1) 
    
    df_ind['dif'] = df_ind['snowdepth_sh'] - df_ind['snowdepth']
    df_ind['dif'] = df_ind['dif'].dt.days - 1
    df_ind = df_ind[df_ind['dif'] > 0]
    
    
    df_stat['snowbreak_count'] = len(df_ind)
    df_stat['snowbreak_max_len'] = df_ind['dif'].max()
    df_stat['snowbreak_mean_len'] = df_ind['dif'].mean()
    df_stat['len_snowbreak'] = df_ind['dif'].sum()         
    
    if len(df_ind) == 0:
        df_stat['snowbreak_max_len'] = [0]
        df_stat['snowbreak_mean_len'] = [0]
        df_stat['len_snowbreak'] = [0]
    
    df_stat['Cv_snowbreak'] = df_ind['dif'].std() / df_ind['dif'].mean()
    
    df_stat['max_snowdepth'] = df_snow['sd'].max()
    df_stat['10D_max_snowdepth'] = df_cut['sd'].rolling('10D').mean().max()
    df_stat['mean_snowdepth_snow_days'] = df_snow['sd'].mean()
    df_stat['sum_snowdepth'] = df_snow['sd'].sum()
    
    try:
        df_stat['mean_snowdepth_snow_period'] = df_cut[df_snow.index[0] : df_snow.index[-1]]['sd'].mean()
        df_stat['Cv_snowdepth_snow_period'] = df_cut[df_snow.index[0] : df_snow.index[-1]]['sd'].std() / df_cut[df_snow.index[0] : df_snow.index[-1]]['sd'].mean()
    except:
        df_stat['mean_snowdepth_snow_period'] = [np.nan]
        df_stat['Cv_snowdepth_snow_period'] = [np.nan]
    
    df_stat['Cv_snowdepth_snow_days'] = df_snow['sd'].std() / df_snow['sd'].mean()
    
    return df_stat
    
def sd_stat_groupby(df_pre):
# =============================================================================
#     В данной функции размечаем наши данные по полю year (задаем кастомный год)
#     Запускаем groupby
#     Меняем индексы в итоговой таблице
# =============================================================================
    df_new = df_pre.copy()
    df_new['year'] = df_new.index.year
    df_new.loc[df_new.index.month >= 8, 'year'] += 1    
     
    df_gr = df_new.groupby(df_new['year']).apply(sd_stat)
               
    df_gr['time'] = pd.date_range(start = str(df_gr.index[0][0]) + '-01-01', freq = 'AS', periods = len(df_gr))

    df_gr['longitude'] = df_pre['longitude'].unique()[0]
    df_gr['latitude'] = df_pre['latitude'].unique()[0]
    
    df_gr = df_gr.set_index(['time', 'latitude', 'longitude'])
    
    return df_gr



if __name__ == '__main__':
# =============================================================================
#     Определение характеристик снежного покрова по данным ERA5
#     Необходимо задать исходный файл с параметром snow_depth и место сохранения результата
# =============================================================================
    path = 'input/'
    input_file = 'snow_depth_full.nc'                               
    
    save_path = 'output/SD_features.nc'                                        
    
    nc_sd = xr.open_dataset(path + input_file)        
    nc_sd = nc_sd['sd'] * 1000
    
    nc_sd.values = xr.where(nc_sd.values > 0.05, nc_sd.values,  0.0)
    
    xarray_list = xarray_to_df(nc_sd)
    
    with mp.Pool(mp.cpu_count()) as p:

        result = list(tqdm(p.imap(sd_stat_groupby, xarray_list[:], chunksize = 1), desc = 'imap', total = len(xarray_list)))
        
        p.close()
        p.join()

    df_full = pd.concat(result)       
    
    xxx = df_full.to_xarray()
    xxx.to_netcdf(save_path)         
       

