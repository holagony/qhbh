# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:18:47 2024

@author: EDY

读取站点数据 温度数据

采暖度日： 采暖度日数=(18'C-平均气温)x 采暖天数 采暖天数固走:10月15日~4月15日（183天）
         平均气温：采暖日的平均气温
采暖日： 当5天滑动平均气温低于或等于5℃时5天中的第一天作为供暖开始日期，即供暖初日。
       当5天滑动平均气温高于或等于5℃时5天中的最后一天作为供暖截止日期，即供暖终日。每个供暖季，自供暖初日到供暖终日为供暖期日数（长度）
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def winter_heating_his(refer_df):
 
    station_id=refer_df['Station_Id_C'].unique()
    
    matched_stations = pd.merge(pd.DataFrame({'Station_Id_C': station_id}),refer_df[['Station_Id_C', 'Station_Name']],on='Station_Id_C')
    matched_stations_unique = matched_stations.drop_duplicates()

    station_name = matched_stations_unique['Station_Name'].values
    station_id=matched_stations_unique['Station_Id_C'].values
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
        data=refer_df[refer_df['Station_Id_C']==name]
        data = data.sort_index()

        tas_rolling = data['TEM_Avg'].rolling(window=5)
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
        
        # if len(data_year) != 0:
        #     data_year['flag']=(data_year['tas_rolling_mean']<=5).astype(int)
        #     first_index = data_year[data_year['flag'] == 1].index.min()
        #     last_index = data_year[data_year['flag'] == 1].index.max()
            
        #     result_start_end.at[n,station_name[idx]]=first_index-timedelta(days=4)
        
        #     if data_year.loc[last_index,'tas_rolling_mean']==0 & last_index.month==4:
        #         result_start_end.at[n,name]=last_index
        #     elif last_index.month !=4  & last_index.day !=30:
        #         result_start_end.at[n,name]=last_index+timedelta(days=1)
        #     result_start_end.at[n,'年']=year-1
        #     n=n+1
        
        
        # 剩余时间
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
                result_hdd18.at[n,name]=np.round((18-data_year.loc[first_index_z:last_index_z,'TEM_Avg'].mean())*183,2)
                result_hdd18.at[n,'年']=year
    
                # 采暖起始日
                result_start_end.at[n,station_name[idx]]=first_index_z.strftime('%Y-%m-%d')
                result_start_end.at[n,name]=last_index_z.strftime('%Y-%m-%d')
                result_start_end.at[n,'年']=year
    
                # 采暖起始日-日序
                result_start_end_num.at[n,station_name[idx]]=(first_index_z-datetime(first_index_z.year, 1, 1)).days+1
                result_start_end_num.at[n,name]=(last_index_z-datetime(last_index_z.year, 1, 1)).days+1
                result_start_end_num.at[n,'年']=year
                
                n=n+1
    
    # #%% 计算信息加入
    # def trend_rate(x):
    #     '''
    #     计算变率（气候倾向率）的pandas apply func
    #     '''
    #     try:
    #         x = x.to_frame()
    #         x['num'] = np.arange(len(x))
    #         x.dropna(how='any', inplace=True)
    #         train_x = x.iloc[:, -1].values.reshape(-1, 1)
    #         train_y = x.iloc[:, 0].values.reshape(-1, 1)
    #         model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
    #         weight = model.coef_[0][0].round(3) * 10
    #         return weight
    #     except:
    #         return np.nan
        
    
    # # 采暖日
    # result_days.at[n,'年']='平均'
    # result_days.loc[n, result_days.columns[1:]] =result_days.iloc[:-1:, 1:].mean(axis=0).round(1)
    # result_days.loc[n+1, result_days.columns[1:]] =result_days.iloc[:-1:, 1:].apply(trend_rate, axis=0)
    # result_days.at[n+1,'年']='变率'
    # result_days.at[n+2,'年']='最大值'
    # result_days.loc[n+2, result_days.columns[1:]] =result_days.iloc[:-3:, 1:].max(axis=0)
    # result_days.at[n+3,'年']='最小值'
    # result_days.loc[n+3, result_days.columns[1:]] =result_days.iloc[:-4:, 1:].min(axis=0)
    
    # result_days['区域均值']=np.round(result_days.iloc[:, 1::].mean(axis=1),1)
    # result_days['区域最大值']=result_days.iloc[:, 1:-1:].max(axis=1)
    # result_days['区域最小值']=result_days.iloc[:, 1:-2:].min(axis=1)    
    
    # # 采暖度日
    # result_hdd18.at[n,'年']='平均'
    # result_hdd18.loc[n, result_hdd18.columns[1:]] =result_hdd18.iloc[:, 1:].mean(axis=0).round(1)
    # result_hdd18.loc[n+1, result_hdd18.columns[1:]] =result_hdd18.iloc[:-1:, 1:].apply(trend_rate, axis=0)
    # result_hdd18.at[n+1,'年']='变率'
    # result_hdd18.at[n+2,'年']='最大值'
    # result_hdd18.loc[n+2, result_hdd18.columns[1:]] =result_hdd18.iloc[:-3:, 1:].max(axis=0)
    # result_hdd18.at[n+3,'年']='最小值'
    # result_hdd18.loc[n+3, result_hdd18.columns[1:]] =result_hdd18.iloc[:-4:, 1:].min(axis=0)
    
    # result_hdd18['区域均值']=np.round(result_hdd18.iloc[:, 1::].mean(axis=1),1)
    # result_hdd18['区域最大值']=result_hdd18.iloc[:, 1:-1:].max(axis=1)
    # result_hdd18['区域最小值']=result_hdd18.iloc[:, 1:-2:].min(axis=1) 
    
    # # 时序
    # result_start_end_num.at[n,'年']='平均'
    # result_start_end_num.loc[n, result_start_end_num.columns[1:]] =result_start_end_num.iloc[1:-1:, 1:].mean(axis=0).round(1)
    # result_start_end_num.loc[n+1, result_start_end_num.columns[1:]] =result_start_end_num.iloc[1:-1:, 1:].apply(trend_rate, axis=0)
    # result_start_end_num.at[n+1,'年']='变率'
    # result_start_end_num.at[n+2,'年']='最大值'
    # result_start_end_num.loc[n+2, result_start_end_num.columns[1:]] =result_start_end_num.iloc[1:-3:, 1:].max(axis=0)
    # result_start_end_num.at[n+3,'年']='最小值'
    # result_start_end_num.loc[n+3, result_start_end_num.columns[1:]] =result_start_end_num.iloc[1:-4:, 1:].min(axis=0)
    
    # result_start_end_num['区域均值1']=np.round(result_start_end_num.iloc[1:, 1::2].mean(axis=1),1)
    # result_start_end_num['区域均值2']=np.round(result_start_end_num.iloc[1:, 2:-1:2].mean(axis=1),1)
    # result_start_end_num.at[0,'区域均值1'] = '开始日期'
    # result_start_end_num.at[0,'区域均值2'] = '结束日期'

    # result_start_end_num['区域最大值1']=result_start_end_num.iloc[1:, 1:-2:2].max(axis=1)
    # result_start_end_num['区域最大值2']=result_start_end_num.iloc[1:, 2:-3:2].max(axis=1)
    # result_start_end_num.at[0,'区域最大值1'] = '开始日期'
    # result_start_end_num.at[0,'区域最大值2'] = '结束日期'
    
    # result_start_end_num['区域最小值1']=result_start_end_num.iloc[1:, 1:-4:2].min(axis=1)     
    # result_start_end_num['区域最小值2']=result_start_end_num.iloc[1:, 2:-5:2].min(axis=1)     
    # result_start_end_num.at[0,'区域最小值1'] = '开始日期'
    # result_start_end_num.at[0,'区域最小值2'] = '结束日期'
    
    
    return result_days,result_hdd18,result_start_end,result_start_end_num


























