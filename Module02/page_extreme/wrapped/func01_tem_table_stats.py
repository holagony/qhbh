# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:43:32 2024

@author: EDY
"""

import pandas as pd
import numpy as np
from Utils.data_processing import data_processing

def tem_table_stats(data_df, time_freq, ele,l_data=None,n_data=None):

    #%% 数据前处理
    if ele == 'DTR':
        data_df['DTR']=data_df['TEM_Max']-data_df['TEM_Min']

    
    #%% 要素匹配
    ele_ment=dict()
    ele_ment['TN10p']='TEM_Min'
    ele_ment['TX10p']='TEM_Max'
    ele_ment['TN90p']='TEM_Min'
    ele_ment['TX90p']='TEM_Max'
    ele_ment['ID']='TEM_Max'
    ele_ment['FD']='TEM_Min'
    ele_ment['TNx']='TEM_Min'
    ele_ment['TXx']='TEM_Max'
    ele_ment['TNn']='TEM_Min'
    ele_ment['TXn']='TEM_Max'
    ele_ment['DTR']='DTR'
    ele_ment['WSDI']='TEM_Max'
    ele_ment['CSDI']='TEM_Max'
    ele_ment['SU']='TEM_Max'
    ele_ment['TR']='TEM_Min'
    ele_ment['GSL']='TEM_Avg'
    ele_ment['high_tem']='TEM_Max'

    
    data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values=ele_ment[ele]) # 统计时段df
    
    #%% 要素计算
    # 冷夜日数 TN10p or 冷昼日数 TX10p
    if ele == 'TN10p' or ele == 'TX10p':
        for i in np.arange(np.size(data_df,1)):
            
            if i==0:
                l_data=l_data/100            
            data_sta=data_df.iloc[:,i]
            data_percentile_10 = data_sta.quantile(l_data)    
            data_df.iloc[:,i] = ((data_df.iloc[:,i] < data_percentile_10)).astype(int)
            
    # 暖夜日数 TN90p or 暖昼日数 TX90p
    elif ele == 'TN90p' or ele == 'TX90p':
        for i in np.arange(np.size(data_df,1)):
            
            if i==0:
                n_data=n_data/100
            data_sta=data_df.iloc[:,i]
            data_percentile_90 = data_sta.quantile(n_data)    
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > data_percentile_90)).astype(int)
            
    # 结冰日数 ID or 霜冻日数 FD
    elif ele == 'ID' or ele == 'FD':
    
        for i in np.arange(np.size(data_df,1)):
            
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] < 0)).astype(int)
            
    # 暖持续指数 WSDI:
    elif ele == 'WSDI':
        
         for i in np.arange(np.size(data_df,1)):
             
             data_sta=data_df.iloc[:,i]
             data_percentile_10 = data_sta.quantile(0.9)  
             data_rolling = data_sta.rolling(window=6)
             data_rolling_min = data_rolling.min().astype(float).round(1)
             data_df.iloc[:,i] = ((data_rolling_min > data_percentile_10)).astype(int)

    # 冷持续指数 CSDI:
    elif ele == 'CSDI':
        
         for i in np.arange(np.size(data_df,1)):
             
             data_sta=data_df.iloc[:,i]
             data_percentile_10 = data_sta.quantile(0.1)  
             data_rolling = data_sta.rolling(window=6)
             data_rolling_min = data_rolling.min().astype(float).round(1)
             data_df.iloc[:,i] = ((data_rolling_min < data_percentile_10)).astype(int)

    # 夏季日数 SU
    elif ele == 'SU':
    
        for i in np.arange(np.size(data_df,1)):
            
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > 25)).astype(int)
    
    # 高温日数       
    elif ele == 'high_tem':
    
        for i in np.arange(np.size(data_df,1)):
            
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > n_data)).astype(int)
            
    # 热夜日数 TR
    elif ele == 'TR':
    
        for i in np.arange(np.size(data_df,1)):
            
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > 20)).astype(int)

    # 生长期长度 GSL:
    elif ele == 'GSL':
        
         for i in np.arange(np.size(data_df,1)):
             
             data_sta=data_df.iloc[:,i]
             data_rolling = data_sta.rolling(window=6)
             data_rolling_min = data_rolling.min().astype(float).round(1)
             data_df.iloc[:,i] = ((data_rolling_min > 5)).astype(int)
                    
    #%% 数据转换
      
    if ele in ['TN10p', 'TX10p', 'TN90p', 'TX90p', 'ID', 'FD', 'SU','TR','GSL','high_tem']:
       
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').sum()
    
        data_df.index = data_df.index.strftime('%Y')

    
    elif ele == 'TNx' or ele == 'TXx' :
        
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').max().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')

    
    elif ele == 'TNn' or ele == 'TXn' :
        
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').min().astype(float).round(1)   
        data_df.index = data_df.index.strftime('%Y')
        
    # elif ele == 'TNn' or ele == 'TXn' :
        
    #     # if time_freq in ['Y','Q']:
            
    #     data_df = data_df.resample('Y').min().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')
        
    if ele in ['DTR','WSDI','CSDI']:
        
        data_df = data_df.resample('Y').mean().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')

    data_df.reset_index(inplace=True)
    data_df.rename(columns={'Datetime': '年'}, inplace=True)
    data_df['年'] = data_df['年'].astype(int)

    
    return data_df
