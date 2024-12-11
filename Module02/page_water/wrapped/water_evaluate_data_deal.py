# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:51:42 2024

@author: EDY

"""
import pandas as pd
from Module02.page_ice.wrapped.func00_data_read_sql import data_read_sql
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2 import sql
from Utils.config import cfg
from Module02.page_water.wrapped.func01_q_stats import stats_q



def water_evaluate_data_deal(train_time, hydro_ids, time_freq, time_freq_data):
    
    table_name = 'qh_climate_other_river_day'
    element_str = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,adnm,Q'
    
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
    
    df=data_read_sql(hydro_ids,element_str,train_time_use,table_name,time_freq)

    df['Datetime'] = pd.to_datetime(df['Datetime'],format='%Y-%m-%d')
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    df['Q'] = df['Q'].apply(lambda x: float(x) if x != '' else np.nan)#.astype(float) # 日尺度
    
    train_data_df = df['Q'].resample('1A').mean().round(1).to_frame()
    train_data_df.index = train_data_df.index.strftime('%Y')
    
    # 按行去取平均
    train_station_data=train_data_df.copy()
    train_station_data=pd.DataFrame(train_data_df.mean(axis=1).round(2))
    train_station_data.columns=[hydro_ids]
    
    return train_data_df, train_station_data
        
    

if __name__=='__main__':
    
    train_time='2023,2024'
    verify_time= '2023,2024'
    sta_ids='40100350'
    time_freq='Y'
    time_freq_data='0'
    train_data_df, train_station_data = water_evaluate_data_deal(train_time,sta_ids,time_freq,time_freq_data)
        
        
        
        
        
        
        
        
        
        
        
        
        
   


