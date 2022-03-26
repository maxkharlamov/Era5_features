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

import warnings



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

def temp_prec_stat(df_cut):
    def linregress(x, y, w=None, b=None):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
    
        if w is None:
            w = np.ones(x.size, dtype=np.float64)
    
        wxy = np.sum(w*y*x)
        wx = np.sum(w*x)
        wy = np.sum(w*y)
        wx2 = np.sum(w*x*x)
        sw = np.sum(w)
    
        den = wx2*sw - wx*wx
    
        if den == 0:
            den = np.finfo(np.float64).eps
    
        if b is None:
            k = (sw*wxy - wx*wy) / den
        else:
            k = (wxy - wx*b) / wx2

        return k

    def last_day_stats(df_cut, date1, date2, label):
        
        last_nday_data = df_cut[date1 : date2]
        
        df_stat['temp_mean_' + label] = [last_nday_data['mean_temp'].mean()]
        df_stat['temp_sum_' + label] = [last_nday_data['mean_temp'].sum()]
        df_stat['frozen_temp_' + label]  = [last_nday_data[last_nday_data['mean_temp'] < 0]['mean_temp'].sum()]
        df_stat['thaw_temp_sum_' + label] = [last_nday_data[last_nday_data['mean_temp'] > 0]['mean_temp'].sum()]
        df_stat['range_temp_' + label] = [last_nday_data['mean_temp'].max() - last_nday_data['mean_temp'].min()]
        df_stat['TP_sum_' + label] = [last_nday_data['prec'].sum()]
        df_stat['SnowTP_sum_' + label] = [last_nday_data[last_nday_data['mean_temp'] < 0]['prec'].sum()]
        df_stat['ThawTP_sum_' + label] = [last_nday_data[last_nday_data['mean_temp'] > 0]['prec'].sum()]
        
        df_stat['temp_Cv_' + label] = [last_nday_data['mean_temp'].std() / last_nday_data['mean_temp'].mean()]
        df_stat['temp_std_' + label] = [last_nday_data['mean_temp'].std()]
        
        df_stat['snowdepth_mean_' + label] = [last_nday_data['snowdepth'].mean()]
        
        df_stat['len_' + label] = [len(last_nday_data)]         # новое
        

        df_stat['temp_trend_coef_' + label] = [linregress(range(len(last_nday_data)),last_nday_data['mean_temp'].values)]   # новое
        
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
    df_stat['frozen_days_1.01'] = [len(df_winter[df_winter['mean_temp']<0][str(df_cut.index.year[-1]) + '-01-01' : max_i])]
    df_stat['frozen_days_15.01'] = [len(df_winter[df_winter['mean_temp']<0][str(df_cut.index.year[-1]) + '-01-15' : max_i])]
    df_stat['frozen_days_1.02'] = [len(df_winter[df_winter['mean_temp']<0][str(df_cut.index.year[-1]) + '-02-01' : max_i])]
    
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
    
    df_stat = last_day_stats(df_cut, max_i, max_i + pd.to_timedelta('10 days'), 'sp_10d')
    df_stat = last_day_stats(df_cut, max_i - pd.to_timedelta('10 days'), max_i + pd.to_timedelta('10 days'), 'bf_10d')
    df_stat = last_day_stats(df_cut, max_i - pd.to_timedelta('10 days'), max_i, 'last_10d')
    
    min_i_sd = df_cut['snowdepth'][df_cut['snowdepth'] > 0].first_valid_index()         
    max_i_sd = df_cut['snowdepth'][df_cut['snowdepth'] > 0].last_valid_index()         
    max_sd_ind = df_cut[df_cut['snowdepth'] == df_cut['snowdepth'].max()].index[-1]     
    
    year_cut = df_cut.index.year[0]
    
    if max_i_sd != None:
        df_stat['TPsum_01.11-SP'] = [df_cut[str(year_cut) + '-11-01' : max_i_sd]['prec'].sum()]
        df_stat['TPsum_01.02-SP10'] = [df_cut[str(year_cut+1) + '-02-01' : max_i_sd + pd.to_timedelta('10 days')]['prec'].sum()]
        df_stat['TPsum_10.02-SP10'] = [df_cut[str(year_cut+1) + '-02-10' : max_i_sd + pd.to_timedelta('10 days')]['prec'].sum()]
        df_stat['TPsum_20.02-SP10'] = [df_cut[str(year_cut+1) + '-02-20' : max_i_sd + pd.to_timedelta('10 days')]['prec'].sum()]
    
    else:
        df_stat['TPsum_01.11-SP'] = [np.nan]
        df_stat['TPsum_01.02-SP10'] = [np.nan]
        df_stat['TPsum_10.02-SP10'] = [np.nan]
        df_stat['TPsum_20.02-SP10'] = [np.nan]
  
    df_sd = df_cut[str(year_cut+1) + '-01-05' : str(year_cut+1) + '-03-31']
   
    df_stat['Smax_05-31.01'] = df_sd[: str(year_cut+1) + '-01-31']['snowdepth'].max()
    df_stat['Smax_05-28.02'] = df_cut[str(year_cut+1) + '-02-05' : str(year_cut+1) + '-02-28']['snowdepth'].max()
    df_stat['Smax_31.01-15.02'] = df_cut[str(year_cut+1) + '-01-31' : str(year_cut+1) + '-02-15']['snowdepth'].max()
    df_stat['Smax_31.01-28.02'] = df_cut[str(year_cut+1) + '-01-31' : str(year_cut+1) + '-02-28']['snowdepth'].max()
    df_stat['Smax_05.01-31.03'] = df_sd['snowdepth'].max()
    
    df_melt = df_cut[max_sd_ind:max_i_sd]
    df_stat['melt_time'] = [len(df_cut[max_sd_ind:max_i_sd])]
    df_stat['melt_time_%'] = [len(df_cut[max_sd_ind:max_i_sd]) / len(df_cut[min_i_sd:max_i_sd]) * 100]
    df_stat['melt_speed'] = df_cut['snowdepth'].max() / df_stat['melt_time']
    
    df_melt = df_cut[max_sd_ind:max_i_sd]
    df_stat['melt_trend_coef'] = [linregress(range(len(df_melt)),df_melt['snowdepth'].values)]      

    df_melt = df_cut[max_i:max_i_sd]
    df_stat['melt_active_trend_coef'] = [linregress(range(len(df_melt)),df_melt['snowdepth'].values)]   
    
    df_stat['melt_time_active'] = [len(df_cut[max_i:max_i_sd])]
    df_stat['melt_time_active_%'] = [len(df_cut[max_i:max_i_sd]) / len(df_cut[min_i_sd:max_i_sd]) * 100]
    df_stat['melt_speed_active'] = df_cut['snowdepth'][max_i:max_i_sd].max() / df_stat['melt_time_active']
    
    df_melt = df_cut[max_sd_ind:max_i]
    df_stat['melt_passive_trend_coef'] = [linregress(range(len(df_melt)),df_melt['snowdepth'].values)]  
        
    df_stat['melt_time_passive'] = [len(df_melt)]
    df_stat['melt_time_passive_%'] = [len(df_melt) / len(df_cut[min_i_sd:max_i_sd]) * 100]
    df_stat['melt_speed_passive'] = df_cut['snowdepth'][max_i:max_i_sd].max() / df_stat['melt_time_passive']

    df_stat = last_day_stats(df_cut, max_sd_ind, max_i_sd, 'sesc_Send')     
    df_stat = last_day_stats(df_cut, max_i_sd,  max_i_sd + pd.to_timedelta('10 days'), 'esc10_10d')  
    
    df_stat = last_day_stats(df_cut, max_sd_ind, max_i, 'Smax_T0')     
    df_stat = last_day_stats(df_cut, max_i, max_i_sd, 'T0_S0')     
    
    
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

    path1 = 'F:/RNF/data_ready_to_use/ERA5/2m_temperature/'                    
    path2 = 'F:/RNF/data_ready_to_use/ERA5/total_precipitation/'
    path3 = 'F:/RNF/data_ready_to_use/ERA5/snow_depth/'
    
    input_file1 = '2m_temperature_full.nc'                         
    input_file2 = 'total_precipitation_full.nc'                    
    input_file3 = 'snow_depth_full.nc'                     
    
    save_path = 'Temp_Prec_SD_full4.nc'                                        
    
    nc_temp = xr.open_dataset(path1 + input_file1)        
    nc_temp = nc_temp['t2m'] - 273.15
    
    nc_pre = xr.open_dataset(path2 + input_file2)        
    nc_pre = nc_pre['tp']*1000
    
    nc_sd = xr.open_dataset(path3 + input_file3)        
    nc_sd = nc_sd['sd'] * 1000
    
    nc_sd.values = xr.where(nc_sd.values > 0.05, nc_sd.values,  0.0)
    
    xarray_list = []
     
    for i in tqdm(range(len(nc_temp ['longitude'][:])), desc = 'make list'):
        for j in range(len(nc_temp ['latitude'][:])):
             
            df_pre = nc_temp[:, j, i].to_dataframe()
            df_pre = pd.concat([df_pre, nc_pre[:, j, i].to_dataframe()['tp']], axis = 1)
            df_pre = pd.concat([df_pre, nc_sd[:, j, i].to_dataframe()['sd']], axis = 1)
            
            xarray_list.append(df_pre)
            
    
    with mp.Pool(mp.cpu_count()) as p:
 
        result = list(tqdm(p.imap(sd_stat_groupby, xarray_list[:], chunksize = 1), desc = 'imap', total = len(xarray_list)))
         
        p.close()
        p.join()
    
 
    df_full = pd.concat(result)       
     
    xxx = df_full.to_xarray()
    xxx.to_netcdf(save_path)         
