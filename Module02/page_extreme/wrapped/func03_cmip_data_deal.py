# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:12:41 2024

@author: EDY
"""

import numpy as np
from Module02.page_energy.wrapped.func06_read_model_data import read_model_data

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


def extreme_pre(ele,data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids,station_dict,l_data=None,n_data=None,GaWIN=None,GaWIN_flag=None,R=None,R_flag=None,RD=None,RD_flag=None,Rxxday=None):
    
    if ele == 'DTR':
        data_a=read_model_data(data_dir,time_scale,insti,scene,'tasmax',stats_times,time_freq,station_ids)
        data_b=read_model_data(data_dir,time_scale,insti,scene,'tasmin',stats_times,time_freq,station_ids)
        data_df=data_a-data_b

    else:
        data_df=read_model_data(data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids)
    
    #%% 要素计算-气温
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
    
    #%% 要素计算-降水
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
            data_df.iloc[:,i] =data_sta_2['result'].astype(float).round(1)

    # 降雨日数 降水强度 大雨日数 中雨日数 特强降水日数
    elif ele in ['RZD','SDII','R25D','R50D','R10D']:
    
        for i in np.arange(np.size(data_df,1)):
                  
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > D[ele])).astype(int)
            
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
      
    if ele in ['TN10p', 'TX10p', 'TN90p', 'TX90p', 'ID', 'FD', 'SU','TR','GSL','RZ','RZD','SDII','R25D','R50D','R10D','R95%D','R95%','R50','R','RD','GaWIN','drought','light_drought','medium_drought','heavy_drought','severe_drought']:
       
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').sum()
    
        data_df.index = data_df.index.strftime('%Y')

    
    elif ele in ['CDD','CWD','Rx1day','Rx5day','Rxxday','TNx','TXx']:
         
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').max().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')

    
    elif ele == 'TNn' or ele == 'TXn' :
        
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').min().astype(float).round(1)   
        data_df.index = data_df.index.strftime('%Y')
        
        
    if ele in ['DTR','WSDI','CSDI']:
        
        data_df = data_df.resample('Y').mean().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')

    data_df.reset_index(inplace=True)
    data_df.rename(columns={'Datetime': '年'}, inplace=True)
    data_df['年'] = data_df['年'].astype(int)
    
    return data_df