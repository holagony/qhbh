# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:07:49 2024

@author: EDY
"""

import pandas as pd
import numpy as np
from Module02.page_ice.wrapped.func00_data_read_sql import data_read_sql
from datetime import  date,datetime, timedelta


def custom_sum(x,method):
    if x.isnull().all():
        return np.nan
    else:
        if method=='sum':
            return np.nansum(x.values)
        elif method=='mean':
            return np.nanmean(x.values)
    
def grass_evaluate_data_deal(element,train_time,sta_ids,time_freq,time_freq_data):


    # 确定表名
    table_dict = dict()
    table_dict['grassland_green_period'] = 'qh_climate_crop_growth'
    table_dict['grassland_yellow_period'] = 'qh_climate_crop_growth'
    table_dict['grassland_growth_period'] = '待定'
    table_dict['grassland_coverage'] = 'qh_climate_grass_cover'
    table_dict['dwei'] = 'qh_climate_grass_yield'  # 草地产量干重
    table_dict['fwei'] = 'qh_climate_grass_yield'  # 草地产量湿重
    table_name = table_dict[element]
    
    # 确定要素
    element_dict = dict()
    element_dict['grassland_green_period'] = 'Crop_Name,GroPer_Name_Ten'
    element_dict['grassland_yellow_period'] = 'Crop_Name,GroPer_Name_Ten'
    element_dict['grassland_growth_period'] = '待定'
    element_dict['grassland_coverage'] = 'Cov'
    element_dict['dwei'] = 'crop_listoc_name,dwei'
    element_dict['fwei'] = 'crop_listoc_name,fwei'
    element_str = element_dict[element]
    elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,' + element_str

    # 4. 读取数据
    # 构建选择时间
    if time_freq == 'Y':
        train_time_use=train_time
    
    elif time_freq == 'Q':# ['%Y,%Y','3,4,5']
        train_time_use=[train_time,time_freq_data]

    elif time_freq== 'M1': #'%Y%m,%Y%m' '%Y,%Y' '%m,%m'
        train_time_use=train_time.split(',')[0]+time_freq_data.split(',')[0]+','+\
                            train_time.split(',')[1]+time_freq_data.split(',')[1]
        
    elif time_freq== 'M2':
        train_time_use=[train_time,time_freq_data]
        
    elif time_freq == 'D1':
        train_time_use= train_time.split(',')[0]+'0101,'+train_time.split(',')[1]+'1231'

    elif time_freq== 'D2': 
        train_time_use=[train_time,time_freq_data]

    train_data=data_read_sql(sta_ids,elements,train_time_use,table_name,time_freq)

    train_data.set_index('Datetime', inplace=True)
    train_data.index = pd.DatetimeIndex(train_data.index)
    train_data['Station_Id_C'] = train_data['Station_Id_C'].astype(str)
    
    if 'Unnamed: 0' in train_data.columns:
        train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    if element in ['dwei','fwei']:
        train_data[element] = train_data[element].astype(float)
        grouped_sum = train_data[['Station_Id_C',element]].groupby(['Station_Id_C', train_data.index]).sum().reset_index()
    
        result_df = grouped_sum.pivot_table(index=['Datetime'], columns=['Station_Id_C'], values=element)  # 统计时段df
        result_df = result_df.resample('Y').apply(custom_sum,'sum')
        result_df.index = result_df.index.strftime('%Y')
  
    elif element in ['grassland_coverage']:
        train_data[element_str] = train_data[element_str].astype(float)
        result_df = train_data.pivot_table(index=train_data.index ,columns=['Station_Id_C'], values=element_str)  # 统计时段df
        result_df = result_df.resample('Y').apply(custom_sum,'mean')


    train_station_data=result_df.copy()

    train_data_df=pd.DataFrame(result_df.mean(axis=1).round(2))
    train_data_df.columns=[element]
    
    return train_data_df,train_station_data
        
    
    
    
#%%
if __name__=='__main__':
    
    element='grassland_coverage'
    train_time='1981,2023'
    sta_ids='52943,56021,56045,56065'
    time_freq='Y'
    time_freq_data='0'
        
        