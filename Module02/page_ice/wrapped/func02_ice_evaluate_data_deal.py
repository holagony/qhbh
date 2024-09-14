# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:51:42 2024

@author: EDY

"""
import pandas as pd
from Module02.page_ice.wrapped.func00_data_read_sql import data_read_sql
import numpy as np
from datetime import  date,datetime, timedelta


def frs_processing(element,df):
    
    if element in ['FRS_DEPTH']:
    # 最大冻结深度
        ele='frs_depth' 
        df.index=df.index+pd.DateOffset(months=-8)
        df = df.pivot_table(index=df.index, columns=['Station_Id_C'], values=ele)  # 参考时段df
        df.replace(999999, np.nan, inplace=True)
    
        df = df.resample('Y').max()

        df.index = df.index.strftime('%Y')
        result_df=df.copy()

        return result_df
    
    if element in ['FRS_START','FRS_END','FRS_TIME']:

        df["时间分组"] = df.index.year - (df.index.month < 9)
        grouped = df.groupby(["Station_Id_C", "时间分组"])
        start_times = grouped.apply(lambda x: x[x["frs_state"] == 2].index.min())
        end_times = grouped.apply(lambda x: x[x["frs_state"] == 2].index.max())
    
        start_times = start_times.reset_index()
        end_times = end_times.reset_index()
    
        start_times.columns = ["站名", "年", "开始时间"]
        end_times.columns = ["站名", "年", "结束时间"]
    
        start_df = start_times.pivot(index="年", columns="站名", values="开始时间")
        end_df = end_times.pivot(index="年", columns="站名", values="结束时间")
        if element in ['FRS_START']:
            result_start=start_df.copy()
            
            result_df=result_start.copy()
            for i in np.arange(np.size(result_start,0)):
                for j in np.arange(np.size(result_start,1)):
                    if pd.notna(result_start.iloc[i,j]):
                        result_df.iloc[i,j]=(result_start.iloc[i,j]-datetime(result_start.iloc[i,j].year, 1, 1)).days+1
                    else:
                        result_df.iloc[i,j]=999999
            result_df[result_df==999999]=np.nan    
            return result_df

        if element in ['FRS_END']:
            result_end=end_df.copy()

            result_df=result_end.copy()
            for i in np.arange(np.size(result_end,0)):
                for j in np.arange(np.size(result_end,1)):
                    if pd.notna(result_end.iloc[i,j]):
                        result_df.iloc[i,j]=(result_end.iloc[i,j]-datetime(result_end.iloc[i,j].year, 1, 1)).days+1
                    else:
                        result_df.iloc[i,j]=999999
            result_df[result_df==999999]=np.nan            
            return result_df

def ice_evaluate_data_deal(element,train_time,verify_time,sta_ids,time_freq,time_freq_data):


    # 3. 确定表名
    table_name = 'qh_climate_cmadaas_day'
    
    if element in ['FRS_DEPTH','FRS_START','FRS_END','FRS_TIME']:
        element_str = 'Station_Id_C,Station_Name,Datetime,frs_1st_top,frs_1st_bot,frs_2nd_top,frs_2nd_bot,frs_state,frs_depth'
        
    elif element in ['SNOW_DEPTH','SNOW_DAYS']:
        element_str = 'Station_Id_C,Station_Name,Datetime,snow_depth'
        
    # 4. 读取数据
    # 构建选择时间
    if time_freq == 'Y':
        if element in ['FRS_DEPTH','FRS_START','FRS_END','FRS_TIME']:
            train_time=train_time.split(',')[0]+','+str(int(train_time.split(',')[1])+1)
            verify_time=verify_time.split(',')[0]+','+str(int(verify_time.split(',')[1])+1)
        
        train_time_use=train_time
        verify_time_use=verify_time
    
    elif time_freq == 'Q':# ['%Y,%Y','3,4,5']
        train_time_use=[train_time,time_freq_data]
        verify_time_use=[verify_time,time_freq_data]

    elif time_freq== 'M2':
        train_time_use=[train_time,time_freq_data]
        verify_time_use=[verify_time,time_freq_data]
        
    elif time_freq == 'D1':
        train_time_use= train_time.split(',')[0]+'0101,'+train_time.split(',')[1]+'1231'
        verify_time_use= verify_time.split(',')[0]+'0101,'+verify_time.split(',')[1]+'1231'

    train_data=data_read_sql(sta_ids,element_str,train_time_use,table_name,time_freq)
    verify_data=data_read_sql(sta_ids,element_str,verify_time_use,table_name,time_freq)

    train_data.set_index('Datetime', inplace=True)
    train_data.index = pd.DatetimeIndex(train_data.index)
    train_data['Station_Id_C'] = train_data['Station_Id_C'].astype(str)
    
    if 'Unnamed: 0' in train_data.columns:
        train_data.drop(['Unnamed: 0'], axis=1, inplace=True)  
        
    verify_data.set_index('Datetime', inplace=True)
    verify_data.index = pd.DatetimeIndex(verify_data.index)
    verify_data['Station_Id_C'] = verify_data['Station_Id_C'].astype(str)

    if 'Unnamed: 0' in verify_data.columns:
        verify_data.drop(['Unnamed: 0'], axis=1, inplace=True)  
        
    if element == 'SNOW_DEPTH':
        ele='snow_depth'
        
        train_data_df = train_data.pivot_table(index=train_data.index, columns=['Station_Id_C'], values=ele) # 统计时段df
        train_data_df = train_data_df.resample('Y').max()
        train_data_df.index = train_data_df.index.strftime('%Y')
        
        verify_data_df = verify_data.pivot_table(index=verify_data.index, columns=['Station_Id_C'], values=ele) # 统计时段df
        verify_data_df = verify_data_df.resample('Y').max()
        verify_data_df.index = verify_data_df.index.strftime('%Y')
        
    elif element == 'SNOW_DAYS':
        ele='num'

        train_data_df=train_data.copy()
        train_data_df['num']=(train_data['snow_depth']>0).astype(int)       
        train_data_df = train_data_df.pivot_table(index=train_data_df.index, columns=['Station_Id_C'], values=ele) # 统计时段df
        train_data_df = train_data_df.resample('Y').sum()
        train_data_df.index = train_data_df.index.strftime('%Y')
        
        verify_data_df=verify_data.copy()
        verify_data_df['num']=(verify_data['snow_depth']>0).astype(int)       
        verify_data_df = verify_data_df.pivot_table(index=verify_data_df.index, columns=['Station_Id_C'], values=ele) # 统计时段df
        verify_data_df = verify_data_df.resample('Y').sum()
        verify_data_df.index = verify_data_df.index.strftime('%Y')
        
    else:
        train_data_df=frs_processing(element,train_data)
        verify_data_df=frs_processing(element,verify_data)
        
    # 按行去取平均
    train_data_df=pd.DataFrame(train_data_df.mean(axis=1).round(2))
    train_data_df.columns=[element]
    
    verify_station_data=verify_data_df.copy()
    
    verify_data_df=pd.DataFrame(verify_data_df.mean(axis=1).round(2))
    verify_data_df.columns=[element]
    
    return train_data_df,verify_data_df,verify_station_data
        
    
    
    
#%%
if __name__=='__main__':
    
    element='SNOW_DEPTH'
    train_time='2020,2021'
    verify_time= '2021,2022'
    sta_ids='51886,52737,52876'
    time_freq='Y'
    time_freq_data='0'
    train_data,verify_data,verify_station_data=ice_evaluate_data_deal(element,train_time,verify_time,sta_ids,time_freq,time_freq_data)
        
        
        
        
        
        
        
        
        
        
        
        
        
   


