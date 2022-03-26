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


def temp_prec_stat(df_cut):
    def last_day_stats(df_cut, backward_days, forward_days, label):
        
        last_nday_data = df_cut[max_i - pd.to_timedelta(str(backward_days) + ' days'):max_i + pd.to_timedelta(str(forward_days) + ' days')]
        
        df_stat['temp_mean_' + label] = [last_nday_data['mean_temp'].mean()]
        df_stat['temp_sum_' + label] = [last_nday_data['mean_temp'].sum()]
        df_stat['frozen_temp_' + label]  = [last_nday_data[last_nday_data['mean_temp'] < 0]['mean_temp'].sum()]
        df_stat['thaw_temp_sum_' + label] = [last_nday_data[last_nday_data['mean_temp'] > 0]['mean_temp'].sum()]
        df_stat['range_temp_' + label] = [last_nday_data['mean_temp'].max() - last_nday_data['mean_temp'].main()]
        df_stat['TP_sum_' + label] = [last_nday_data['prec'].sum()]
        df_stat['SnowTP_sum_' + label] = [last_nday_data[last_nday_data['mean_temp'] < 0]['prec'].sum()]
        df_stat['ThawTP_sum_' + label] = [last_nday_data[last_nday_data['mean_temp'] > 0]['prec'].sum()]
        
        df_stat['temp_Cv_' + label] = [last_nday_data['mean_temp'].std() / last_nday_data['mean_temp'].mean()]
        df_stat['temp_std_' + label] = [last_nday_data['mean_temp'].std()]
        return df_stat
    
    
    df_cut = df_cut.rename(columns = {'t2m':'mean_temp', 'tp':'prec', 'sd' : 'snowdepth'})
    df_stat = pd.DataFrame()
    
    max_i = df_cut[df_cut['mean_temp'] < 0].index.max()
    min_i = df_cut.index[df_cut['mean_temp'] < 0].min()
    
    df_winter = df_cut[min_i:max_i]
    
    df_stat['frozen_days'] = [len(df_winter[df_winter['mean_temp'] < 0])]
    df_stat['winter_days'] = [len(df_winter)]
    df_stat['subzero_temp_sum'] = [df_winter[df_winter['mean_temp'] < 0]['mean_temp'].sum()]
    df_stat['subzero_temp_mean'] = [df_winter[df_winter['mean_temp'] < 0]['mean_temp'].mean()]
    df_stat['winter_temp_mean'] = [df_winter['mean_temp'].mean()]
    df_stat['winter_temp_sum'] = [df_winter['mean_temp'].mean()]
    df_stat['thaw_temp_sum'] = [df_winter[df_winter['mean_temp'] > 0]['mean_temp'].sum()]
    df_stat['thaw_temp_mean'] = [df_winter[df_winter['mean_temp'] > 0]['mean_temp'].mean()]
    
    uzp = df_winter[['mean_temp', 'prec']].copy()          
    uzp['mean_temp'] = np.where(uzp['mean_temp'].values < 0, 1, np.nan)
    uzp = uzp['mean_temp'].rolling(5).mean()        
    min_i_uzp = uzp.first_valid_index() 
    max_i_uzp = uzp.last_valid_index() 
    
    df_uzp = df_winter[min_i_uzp:]  
    
    df_stat['frozen_days_uzp'] = [len(df_uzp[df_uzp['mean_temp']<0])]
    df_stat['frozen_days_1.01'] = [len(df_winter[df_winter['mean_temp']<0][str(df_winter.index.year[-1]) + '-01-01' : max_i])]
    df_stat['frozen_days_15.01'] = [len(df_winter[df_winter['mean_temp']<0][str(df_winter.index.year[-1]) + '-01-15' : max_i])]
    df_stat['frozen_days_1.02'] = [len(df_winter[df_winter['mean_temp']<0][str(df_winter.index.year[-1]) + '-02-01' : max_i])]
    
    df_winter_1 = df_winter[df_winter['mean_temp'] < 0]
    
    df_ind = pd.DataFrame()
    df_ind['mean_temp'] = df_winter_1.index
    
    aaa = pd.DataFrame()    
    aaa['mean_temp_sh'] = df_winter_1['mean_temp'].shift(1).dropna().index
    
    df_ind = pd.concat([df_ind, aaa], axis = 1) 
    
    df_ind['dif'] = df_ind['mean_temp_sh'] - df_ind['mean_temp']
    df_ind['dif'] = df_ind['dif'].dt.days - 1
    df_ind = df_ind[df_ind['dif'] > 0]

    df_ind['thaw_temp_mean_per_thaw'] = [df_cut[df_ind['mean_temp'].iloc[i]:df_ind['mean_temp_sh'].iloc[i]]['mean_temp'].values[1:-1].mean() for i in range(len(df_ind))]
    df_ind['ThawTP_sum_per_thaw'] = [df_cut[df_ind['mean_temp'].iloc[i]:df_ind['mean_temp_sh'].iloc[i]]['prec'].values[1:-1].sum() for i in range(len(df_ind))]
    

    df_stat['thaw_count'] = [len(df_ind)]                
    df_stat['thaw_days'] = [len(df_winter[df_winter['mean_temp'] > 0]['mean_temp'])]
    
    df_stat['thaw_mean_len'] = df_stat['thaw_days'] / df_stat['thaw_count']
    df_stat['thaw_max_len'] = [df_ind['dif'].max()] 
    
    df_stat['thaw_temp_mean_per_thaw'] = df_ind['thaw_temp_mean_per_thaw'].mean()       
    df_stat['ThawTP_sum_per_thaw'] = df_ind['ThawTP_sum_per_thaw'].mean()             

    if  len(df_winter) != 0:
    
        df_stat['thaw_days_Cv'] = [df_ind['dif'].std() / df_ind['dif'].mean()]
        df_stat['winter_temp_Cv'] = [df_winter['mean_temp'].std() / df_winter['mean_temp'].mean()]
        df_stat['subzero_temp_Cv'] = [df_winter[df_winter['mean_temp'] < 0]['mean_temp'].std() / df_winter[df_winter['mean_temp'] < 0]['mean_temp'].mean()]
        df_stat['thaw_temp_Cv'] = [df_winter[df_winter['mean_temp'] > 0]['mean_temp'].std() / df_winter[df_winter['mean_temp'] > 0]['mean_temp'].mean()]
        
        df_stat['SnowTP_Cv'] = [df_winter[df_winter['mean_temp'] < 0]['prec'].std() / df_winter[df_winter['mean_temp'] < 0]['prec'].mean()]
        df_stat['ThawTP_Cv'] = [df_winter[df_winter['mean_temp'] > 0]['prec'].std() / df_winter[df_winter['mean_temp'] > 0]['prec'].mean()]
    else:
        df_stat['thaw_days_Cv'] = [np.nan]        
        df_stat['winter_temp_Cv'] = [np.nan]
        df_stat['subzero_temp_Cv'] = [np.nan]
        df_stat['thaw_temp_Cv'] = [np.nan]
        
        df_stat['SnowTP_Cv'] = [np.nan]
        df_stat['ThawTP_Cv'] = [np.nan]
        
    
    df_stat['SnowTP'] = [df_winter[df_winter['mean_temp'] < 0]['prec'].sum()]
    df_stat['ThawTP'] = [df_winter[df_winter['mean_temp'] > 0]['prec'].sum()]
    
    df_stat = last_day_stats(df_cut, 0, 10, 'sp_10d')
    df_stat = last_day_stats(df_cut, 10, 10, 'bf_10d')
    df_stat = last_day_stats(df_cut, 10, 0, 'last_10d')
    
    len_ex = len(df_cut[df_cut.index.month.isin([11,12,1,2,3])]['mean_temp'].dropna())
        
    if len_ex/151 < 0.9:            
        df_stat = df_stat * np.nan
        
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
     
    df_gr = df_new.groupby(df_new['year']).apply(temp_prec_stat)
               
    df_gr['time'] = pd.date_range(start = str(df_gr.index[0][0]) + '-01-01', freq = 'AS', periods = len(df_gr))

    df_gr['longitude'] = df_pre['longitude'].unique()[0]
    df_gr['latitude'] = df_pre['latitude'].unique()[0]
    
    df_gr = df_gr.set_index(['time', 'latitude', 'longitude'])
    
    return df_gr

if __name__ == '__main__':
# =============================================================================
#     Определение характеристик зимнего периода по осадкам и температурам ERA5
#     Необходимо задать исходные файлы с данными prec и temp и место сохранения результата
# =============================================================================

    
    path1 = 'I:/RNF/data_ready_to_use/ERA5/2m_temperature/'                    
    path2 = 'I:/RNF/data_ready_to_use/ERA5/total_precipitation/'
    
    input_file1 = '2m_temperature_full.nc'                         
    input_file2 = 'total_precipitation_full.nc'                    
    
    save_path = 'Temp_Prec_full.nc'                                        
    
    nc_temp = xr.open_dataset(path1 + input_file1)        
    nc_temp = nc_temp['t2m'] - 273.15 
    
    nc_pre = xr.open_dataset(path2 + input_file2)        
    nc_pre = nc_pre['tp']*1000 
    
    
    xarray_list = []
     
    for i in tqdm(range(len(nc_temp ['longitude'][:])), desc = 'make list'):
        for j in range(len(nc_temp ['latitude'][:])):
             
            df_pre = nc_temp[:, j, i].to_dataframe()
            df_pre = pd.concat([df_pre, nc_pre[:, j, i].to_dataframe()['tp']], axis = 1)
             
            xarray_list.append(df_pre)
            

    with mp.Pool(mp.cpu_count()) as p:
 
        result = list(tqdm(p.imap(sd_stat_groupby, xarray_list[:], chunksize = 1), desc = 'imap', total = len(xarray_list)))
         
        p.close()
        p.join()
    
 
    df_full = pd.concat(result)       
     
    xxx = df_full.to_xarray()
    xxx.to_netcdf(save_path)         
