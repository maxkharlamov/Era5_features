# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:31:03 2020

@author: kharl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

import multiprocessing as mp
import xarray as xr

def nc_concat (path_input, folder, save_path):
    
    
    files_len = len([x for x in os.listdir(path_input + os.sep + folder)])
    
    files = [x for x in os.listdir(path_input + os.sep + folder)]
    xr_list = []
    for i in tqdm(range(files_len), desc = folder):
        
        path = path_input + os.sep + folder + os.sep + files[i]
        
        concat_dataset = xr.open_dataset(path)
        if folder == 'total_precipitation':
            concat_dataset = concat_dataset.resample(time = '1D').sum()
        else:
            concat_dataset = concat_dataset.resample(time = '1D').mean()             
        xr_list.append(concat_dataset)
        
    concat_dataset = xr.concat(xr_list, dim='time')           
    concat_dataset.to_netcdf(save_path + folder + '_full.nc')

path_input = 'D:/RNF/data_download/ERA5/'
save_path = 'D:/RNF/ready_to_use/ERA5/'

fldrs = ["2m_temperature", 'total_precipitation']

for folder in fldrs[:]:
    nc_concat(path_input, folder, save_path)
   
