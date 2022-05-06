# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:41:34 2019

@author: kharl
"""
from tqdm import tqdm

from itertools import product

from multiprocessing import Pool
import multiprocessing as mp
import cdsapi


def era5_download_new(year, variable, area):
    
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        #'reanalysis-era5-single-levels-preliminary-back-extension',
        {
            'product_type':'reanalysis',
            'format':'netcdf',
            'variable': variable,
            'year': year,
            'area': area, # N, W, S, E
            'month':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12'
            ],
            'day':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12',
                '13','14','15',
                '16','17','18',
                '19','20','21',
                '22','23','24',
                '25','26','27',
                '28','29','30',
                '31'
            ],
            'grid' : '0.25/0.25',
            'time':[
                '00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00','07:00','08:00',
                '09:00','10:00','11:00',
                '12:00','13:00','14:00',
                '15:00','16:00','17:00',
                '18:00','19:00','20:00',
                '21:00','22:00','23:00'
            ]
        },

        'Era5_' + variable + '_' + year + '.nc')
    

if __name__ == '__main__':
    
    
    years = [str(y) for y in range(1979,2023)]  
    
    
    variables = [ '2m_temperature', 'snow_depth', 'total_precipitation',
                 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
                 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
                 'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3', 'soil_temperature_level_4']
    area = [70, 0, 40, 60]    
    
    with mp.Pool(16) as p:
        result = p.starmap(era5_download_new, product(years, variables, area))         
        p.close()
        p.join()
    
    
       
