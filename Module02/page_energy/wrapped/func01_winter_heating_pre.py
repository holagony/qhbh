# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:39:33 2024

@author: EDY
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Module02.page_energy.wrapped.func06_read_model_data import read_model_data



def winter_heating_pre(element,data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids,station_dict):
    
    refer_df=read_model_data(data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids)
    
    station_id=refer_df.columns
    
    matched_stations = pd.merge(pd.DataFrame({'站号': station_id}),station_dict[['站号', '站名']],on='站号')
    matched_stations_unique = matched_stations.drop_duplicates()

    station_name = matched_stations_unique['站名'].values
    station_id=matched_stations_unique['站号'].values
    #%% 五天滑动平均求开始和结束
    time_year=refer_df.index.year.unique()
    
    # 创建保存array
    station_pairs = np.array([(name, id) for name, id in zip(station_name, station_id)]).ravel()
    station_pairs_list = station_pairs.tolist()
    column_names = ['年'] + station_pairs_list
    result_start_end = pd.DataFrame(columns=column_names)
    result_start_end_num = pd.DataFrame(columns=column_names)
    result_days = pd.DataFrame(columns=['年'] + station_id.tolist())
    result_hdd18 = pd.DataFrame(columns=['年'] + station_id.tolist())
    
    for idx,name in enumerate(station_id):
        n=0
        # break
        data=refer_df[name].to_frame()
        data = data.sort_index()

        tas_rolling = data[name].rolling(window=5)
        data['tas_rolling_mean'] = tas_rolling.mean()
        data.dropna(inplace=True)
        
        result_start_end.at[n,name]='结束时间'
        result_start_end.at[n,station_name[idx]]='开始时间'
        
        result_start_end_num.at[n,name]='结束时间'
        result_start_end_num.at[n,station_name[idx]]='开始时间'
        n=n+1
        
        # 第一个时间点的前半年
        year = time_year[0]
        start_date = pd.Timestamp(year=year-1, month=10, day=5)
        end_date = pd.Timestamp(year=year, month=4, day=30)
    
        data_year = data[(data.index >= start_date) & (data.index <= end_date)]
        
        for year in time_year[:-1:]:
            # break
            data_year = data[((data.index.year == year) & (data.index.month >= 10)) |
                             ((data.index.year == year+1) & (data.index.month <= 4))]
        
            start_date = pd.Timestamp(year=year, month=10, day=5)
            end_date = pd.Timestamp(year=year+1, month=4, day=30)
            data_year = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if len(data_year) != 0:
    
                data_year['flag']=(data_year['tas_rolling_mean']<=5).astype(int)
        
                first_index = data_year[data_year['flag'] == 1].index.min()
                last_index = data_year[data_year['flag'] == 1].index.max()
            
                first_index_z=first_index-timedelta(days=4)
        
                if data_year.loc[last_index,'tas_rolling_mean']==0:
                    last_index_z=last_index 
                elif last_index.month !=4  & last_index.day !=30:
                    last_index_z=last_index+timedelta(days=1)
                    
                else:
                    last_index_z=last_index 
    
                
                # 采暖日
                result_days.at[n,name]=(last_index_z- first_index_z).days
                result_days.at[n,'年']=year
    
                # 采暖度日
                result_hdd18.at[n,name]=np.round((18-data_year.loc[first_index_z:last_index_z,name].mean())*183,2)
                result_hdd18.at[n,'年']=year
    
                # 采暖起始日
                result_start_end.at[n,station_name[idx]]=first_index_z.strftime('%Y-%m-%d')
                result_start_end.at[n,name]=last_index_z.strftime('%Y-%m-%d')
                result_start_end.at[n,'年']=int(year)
        
                    # 采暖起始日-日序
                result_start_end_num.at[n,station_name[idx]]=(first_index_z-datetime(first_index_z.year, 1, 1)).days+1
                result_start_end_num.at[n,name]=(last_index_z-datetime(last_index_z.year, 1, 1)).days+1
                result_start_end_num.at[n,'年']=int(year)
                
                n=n+1
    
    return result_days,result_hdd18,result_start_end,result_start_end_num


    