# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:37:05 2024

@author: EDY

"""

import pandas as pd
import numpy as np
from Utils.data_processing import data_processing

def persistent_time(df,time_freq):
    
    df['group'] = (df.iloc[:,0] != df.iloc[:,0].shift()).cumsum()
    
    if time_freq in ['Y','Q']:
        df['year'] = df.index.year            
        df['group'] = df['year'].astype(str) + '_' + df['group'].astype(str) 
    elif time_freq in ['M1','M2']:
        df['month'] = df.index.month 
        df['year'] = df.index.year            
        df['group'] = df['year'].astype(str) + '_' + df['month'].astype(str)+ '_' + df['group'].astype(str) 
           
    group_sums = df.groupby('group')[df.columns[0]].sum()
    last_ones = df[df[df.columns[0]] == 1].groupby('group').last().index
    group_sum_dict = group_sums.to_dict()
    df['result'] = 0
    for group_id in last_ones:
        last_one_index = df[df['group'] == group_id].index[-1]
        df.at[last_one_index, 'result'] = group_sum_dict[group_id]
        
    return df
                

def pre_table_stats(data_df, time_freq, ele,GaWIN=None,GaWIN_flag=None,R=None,R_flag=None,RD=None,RD_flag=None,Rxxday=None):

    if ele =='GaWIN':
        data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values='win_s_2mi_avg') # 统计时段df
    else:
        data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values='PRE_Time_2020') # 统计时段df
    
    D=dict()
    D['RZD']=0
    D['SDII']=1
    D['R25D']=25
    D['R10D']=10
    D['R50D']=50

    #%% 要素计算
    # 持续干期 CDD
    if ele == 'CDD':
        for i in np.arange(np.size(data_df,1)):
            
            data_sta=data_df.iloc[:,i].to_frame()
            data_sta.columns = data_sta.columns.get_level_values(0)
            data_sta_1= ((data_sta == 0)).astype(int)
            data_sta_2=persistent_time(data_sta_1,time_freq)
            data_df.iloc[:,i] =data_sta_2['result']

                        
    # 持续湿期 CWD
    if ele == 'CWD':
        for i in np.arange(np.size(data_df,1)):
            
            data_sta=data_df.iloc[:,i].to_frame()
            data_sta.columns = data_sta.columns.get_level_values(0)
            data_sta_1= ((data_sta > 0)).astype(int)
            data_sta_2=persistent_time(data_sta_1,time_freq)
            data_df.iloc[:,i] =data_sta_2['result']

    # 降雨日数 降水强度 大雨日数 中雨日数 特强降水日数
    elif ele in ['RZD','SDII','R25D','R50D','R10D']:
    
        for i in np.arange(np.size(data_df,1)):
                  
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] >= D[ele])).astype(int)
            
    # 特强降水
    elif ele =='R50':
    
        for i in np.arange(np.size(data_df,1)):
            data_sta=data_df.iloc[:,i]
            data_percentile_90 = data_sta.quantile(0.95) 
            data_df.iloc[((data_df.iloc[:,i] < data_percentile_90)),i] = 0

    # 强降水
    elif ele =='R95%':
    
        for i in np.arange(np.size(data_df,1)):
            
            data_df.iloc[((data_df.iloc[:,i] < 50)),i] = 0
            
    # 强降水日数 R95D
    elif ele == 'R95%D':
        for i in np.arange(np.size(data_df,1)):
            
            data_sta=data_df.iloc[:,i]
            data_percentile_90 = data_sta.quantile(0.95)    
            data_df.iloc[:,i] = ((data_df.iloc[:,i] >= data_percentile_90)).astype(int)
    # 5日最大降水 Rx5day:
    elif ele == 'Rx5day':
        
         for i in np.arange(np.size(data_df,1)):
                    
             data_sta=data_df.iloc[:,i]
             data_rolling = data_sta.rolling(window=5)
             data_rolling_sum = data_rolling.sum()
             data_df.iloc[:,i] = data_rolling_sum
             
    # 自定义风
    elif ele =='GaWIN':
    
        for i in np.arange(np.size(data_df,1)):
            
            if GaWIN_flag==1:
                data_df.iloc[((data_df.iloc[:,i] <= GaWIN)),i] = 0
            elif GaWIN_flag==2:
                data_df.iloc[((data_df.iloc[:,i] < GaWIN)),i] = 0
            elif GaWIN_flag==3:
                data_df.iloc[((data_df.iloc[:,i] > GaWIN)),i] = 0
            elif GaWIN_flag==4:
                data_df.iloc[((data_df.iloc[:,i] >= GaWIN)),i] = 0
                
    # 自定义降水
    elif ele =='R':
    
        for i in np.arange(np.size(data_df,1)):
            
            if R_flag==1:
                data_df.iloc[((data_df.iloc[:,i] <= R)),i] = 0
            elif R_flag==2:
                data_df.iloc[((data_df.iloc[:,i] < R)),i] = 0
            elif R_flag==3:
                data_df.iloc[((data_df.iloc[:,i] > R)),i] = 0
            elif R_flag==4:
                data_df.iloc[((data_df.iloc[:,i] >= R)),i] = 0
                
    # 自定义降水日
    elif ele =='RD':
    
        for i in np.arange(np.size(data_df,1)):
            
            if RD_flag==1:
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] > RD)).astype(int)
            elif RD_flag==2:
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] >= RD)).astype(int)
                
            elif RD_flag==3:
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] <= RD)).astype(int)

            elif RD_flag==4:
  
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] < RD)).astype(int)
                                    
    # x日最大降水 Rxxday:
    elif ele == 'Rxxday':
        
         for i in np.arange(np.size(data_df,1)):
             
             data_sta=data_df.iloc[:,i]
             data_rolling = data_sta.rolling(window=Rxxday)
             data_rolling_sum = data_rolling.sum()
             data_df.iloc[:,i] = data_rolling_sum
             
    #%% 数据转换
      
    if ele in ['RZ','RZD','SDII','R25D','R50D','R10D','R95%D','R95%','R50','R','RD','GaWIN']:
       
        data_df = data_df.resample('Y').sum()
        data_df.index = data_df.index.strftime('%Y')

    elif ele in ['CDD','CWD','Rx1day','Rx5day','Rxxday']:
                    
        data_df = data_df.resample('Y').max()
        data_df.index = data_df.index.strftime('%Y')

    
    data_df.reset_index(inplace=True)
    data_df.rename(columns={'Datetime': '年'}, inplace=True)
    data_df['年'] = data_df['年'].astype('int64')

    
    return data_df
