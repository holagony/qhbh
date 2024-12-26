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

def frs_processing(element,df):
    
    if element in ['FRS_DEPTH']:
    # 最大冻结深度
        ele='frs_depth' 
        #df.index=df.index+pd.DateOffset(months=-8)
        df = df.pivot_table(index=df.index, columns=['Station_Id_C'], values=ele)  # 参考时段df
        df.replace(999999, np.nan, inplace=True)
    
        # df = df.resample('Y').max()
        df = df.resample(rule='AS-SEP').max()
        df.index = df.index.strftime('%Y')
        result_df=df.copy()

        return result_df
    
    if element in ['FRS_START','FRS_END','FRS_TIME']:

        df["时间分组"] = df.index.year - (df.index.month < 9)
        df = df.resample(rule='AS-SEP').max()
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

            
        if element in ['FRS_TIME']:
            data_len_df=end_df.copy()
            for i in np.arange(np.size(start_df,0)):
                for j in np.arange(np.size(start_df,1)):
                    if pd.notna(start_df.iloc[i,j]) & pd.notna(end_df.iloc[i,j]):
                        data_len_df.iloc[i,j]=(end_df.iloc[i,j]-start_df.iloc[i,j]).days
                    else:
                        data_len_df.iloc[i,j]=999999
            data_len_df[data_len_df==999999]=np.nan    
            result_df=data_len_df.copy()
            
            return result_df

def ice_evaluate_data_deal(element,train_time,sta_ids,time_freq,time_freq_data):

    if element in ['FRS_DEPTH', 'FRS_START', 'FRS_END', 'FRS_TIME','SNOW_DEPTH', 'SNOW_DAYS']:

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
    
     
        train_data=data_read_sql(sta_ids,element_str,train_time_use,table_name,time_freq)
    
        train_data.set_index('Datetime', inplace=True)
        train_data.index = pd.DatetimeIndex(train_data.index)
        train_data['Station_Id_C'] = train_data['Station_Id_C'].astype(str)
        
        if 'Unnamed: 0' in train_data.columns:
            train_data.drop(['Unnamed: 0'], axis=1, inplace=True)  
            
        if element == 'SNOW_DEPTH':
            ele='snow_depth'
            
            train_data_df = train_data.pivot_table(index=train_data.index, columns=['Station_Id_C'], values=ele) # 统计时段df
            train_data_df = train_data_df.resample('Y').max()
            train_data_df.index = train_data_df.index.strftime('%Y')
            
    
        elif element == 'SNOW_DAYS':
            ele='num'
    
            train_data_df=train_data.copy()
            train_data_df['num']=(train_data['snow_depth']>0).astype(int)       
            train_data_df = train_data_df.pivot_table(index=train_data_df.index, columns=['Station_Id_C'], values=ele) # 统计时段df
            train_data_df = train_data_df.resample('Y').sum()
            train_data_df.index = train_data_df.index.strftime('%Y')
            
      
        else:
            train_data_df=frs_processing(element,train_data)
            
        # 按行去取平均
        train_station_data=train_data_df.copy()
    
        train_data_df=pd.DataFrame(train_data_df.mean(axis=1).round(2))
        train_data_df.columns=[element]
    elif element in ['ICE_AREA', 'ICE_RESERVES']:
        
        def get_bingchuan_data(sta_ids,element_str,stats_times):
            sta_ids = tuple(sta_ids.split(','))

            conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
            cur = conn.cursor()
            
            elements = 'datatime,zone,bcname,bcid,' + element_str
    
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.qh_climate_bingchuan
                            WHERE
                                CAST(SUBSTRING(datatime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND bcid IN %s
                            """)
    
            # 根据sql获取统计年份data
            start_year = stats_times.split(',')[0]
            end_year = stats_times.split(',')[1]
            cur.execute(query, (start_year, end_year, sta_ids))
            data = cur.fetchall()
    
            df = pd.DataFrame(data)
            df.columns = elements.split(',')
            df[element_str] = df[element_str].astype(float).apply(lambda x: np.nan if x > 9999 else x)
            
            df.rename(columns={'bcid': 'Station_Id_C'}, inplace=True)
            df.rename(columns={'datatime': 'Datetime'}, inplace=True)

            df.set_index('Datetime', inplace=True)
            df.index =pd.to_datetime(df.index)
            
            cur.close()
            conn.close()
            
            return df
        
        element_str=dict()
        element_str['ICE_AREA']='area'
        element_str['ICE_RESERVES']='bcid'
        
        data_df = get_bingchuan_data(sta_ids, element_str[element], train_time)
        data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values=element_str[element])  # 近1年df
        data_df.index = data_df.index.strftime('%Y')
        
        train_station_data=data_df.copy()
        train_data_df=data_df.copy()

        train_data_df=pd.DataFrame(train_data_df.mean(axis=1).round(2))
        train_data_df.columns=[element]
        
    return train_data_df,train_station_data
        
    
    
    
#%%
if __name__=='__main__':
    
    element='ICE_AREA'
    train_time='2020,2021'
    verify_time= '2021,2022'
    sta_ids='3020201'
    time_freq='Y'
    time_freq_data='0'
    train_data_df,train_station_data=ice_evaluate_data_deal(element,train_time,sta_ids,time_freq,time_freq_data)
        
        
        
        
        
        
        
        
        
        
        
        
        
   


