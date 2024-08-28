# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:49:08 2024

读取nc数据，插值到指定经纬度站点，使用邻近插值法

温度数据

采暖度日： 采暖度日数=(18'C-平均气温)x 采暖天数 采暖天数固走:10月15日~4月15日（183天）
         平均气温：采暖日的平均气温
采暖日： 当5天滑动平均气温低于或等于5℃时5天中的第一天作为供暖开始日期，即供暖初日。
       当5天滑动平均气温高于或等于5℃时5天中的最后一天作为供暖截止日期，即供暖终日。每个供暖季，自供暖初日到供暖终日为供暖期日数（长度）
"""

import netCDF4 as nc
from datetime import  date,datetime, timedelta
import numpy as np
import pandas as pd
from Utils.config import cfg
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression

def increment_date(start_date, days):
    """
    不考虑闰年的计算
    """
    # 月份的天数，不考虑闰年
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # 计算新的日期
    current_month = start_date.month
    current_year = start_date.year
    current_day = start_date.day + days
    
    while current_day > days_per_month[current_month - 1]:
        current_day -= days_per_month[current_month - 1]
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    return date(current_year, current_month, current_day)


def time_choose(time_freq,stats_times,dates):        


    if time_freq== 'Y':
        # Y
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        
        date_indices = [i for i, date in enumerate(dates) if int(start_year) <= date.year <= int(end_year)]
    
    elif time_freq== 'Q':

        # Q
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        month = stats_times[1]
        month = list(map(int,month.split(',')))
        
       
        date_indices = [i for i, date in enumerate(dates) if ((int(start_year) <= date.year <= int(end_year)) & (date.month in month))]
    elif time_freq== 'M1':
   
        # M1
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        
        date_indices = [i for i, date in enumerate(dates) if ((int(start_time[:4:]) <= date.year <= int(end_time[:4:])) 
                                                              & (int(start_time[5::]) <= date.month <= int(end_time[5::])))]
    elif time_freq== 'M2':
    
        # M2
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        month = stats_times[1]
        month = list(map(int,month.split(',')))
        
        date_indices = [i for i, date in enumerate(dates) if ((int(start_year) <= date.year <= int(end_year)) & (date.month in month))]
    elif time_freq== 'D1':
    
        # D1
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        
        start_date_nc_object = datetime.strptime(start_time, '%Y%m%d')
        end_date_nc_object = datetime.strptime(end_time, '%Y%m%d')
        date_indices = [i for i, date in enumerate(dates) if start_date_nc_object <= date <= end_date_nc_object]
    
    elif time_freq== 'D2':
    
    # D2
    
        def is_date_within_range(date_month,date_day, start_month, start_day, end_month, end_day):
            input_date = datetime(2000, date_month,date_day)
            
            start_date = datetime(2000, start_month, start_day)
            end_date = datetime(2000, end_month, end_day)
            
            return start_date <= input_date <= end_date
        
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        month = stats_times[1]
        month = month.split(',')
        start_month=month[0]
        end_month=month[1]
        
        date_indices = [i for i, date in enumerate(dates) if ((int(start_year) <= date.year <= int(end_year)) 
                                                              & (is_date_within_range(date.month,date.day, 
                        int(start_month[:2]), int(start_month[2:]), int(end_month[:2]), int(end_month[2:]))))]
        
    return date_indices


def winter_heating_pre(tas_paths,station_id,time_freq,stats_times):
    #%% 站点编号 和 站点名经纬度匹配
    df_station=pd.read_csv(cfg.FILES.STATION,encoding='gbk')
    df_station['区站号']=df_station['区站号'].astype(str)
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '站点名']],on='区站号')
    station_name = matched_stations['站点名'].values
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '经度']],on='区站号')
    target_lons = matched_stations['经度'].values
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '纬度']],on='区站号')
    target_lats = matched_stations['纬度'].values
    
    # 初始化 DataFrame
    df = pd.DataFrame(columns=['datetime', 'station_id_c', 'lon', 'lat', 'tas'])
    
    # 参考日期
    
    # 遍历所有文件
    df = pd.DataFrame(columns=['datetime', 'station_id_c', 'lon', 'lat', 'tas'])
    for tas_path in tas_paths:
        # break
        with nc.Dataset(tas_path) as tas_dataset:
            lon = tas_dataset.variables['lon'][:]
            lat = tas_dataset.variables['lat'][:]
            tas = tas_dataset.variables['tas'][:]
            time = tas_dataset.variables['time'][:]
            time_var = tas_dataset.variables['time']
            time_calendar=time_var.calendar
            time_units = time_var.units
            ref_time_str = time_units.split('since ')[1]
    
        ref_date = datetime(int(ref_time_str[:4:]), int(ref_time_str[5:7:]),int(ref_time_str[8:10:]))
    
        lons, lats = np.meshgrid(lon, lat)
        
        if time_calendar=='365_day':
            dates = [increment_date(ref_date, int(t)) for t in time[:]]
        else:
            dates = [ref_date + timedelta(days=int(t)) for t in time[:]]
        
        date_indices = time_choose(time_freq,stats_times,dates)
    
        tas = tas[date_indices,:, :]
        dates = [dates[i] for i in date_indices]
        
        points = np.vstack((lons.flatten(), lats.flatten())).T
        
        interpolated_data = np.zeros((len(dates) * len(target_lons),))
        
        for i in range(len(dates)):
            values = tas[i, :, :].flatten()
            interpolated_data[i * len(target_lons):(i + 1) * len(target_lons)] = griddata(points, values, (target_lons, target_lats), method='nearest')
        
        dates_repeated = np.repeat(dates, len(target_lons))
        
        station_id_repeated = np.tile(station_id, len(dates))
        
        # 创建一个新的DataFrame来存储当前循环的数据
        new_df = pd.DataFrame({
            'datetime': dates_repeated,
            'station_id_c': station_id_repeated,
            'lon': np.tile(target_lons, len(dates)),
            'lat': np.tile(target_lats, len(dates)),
            'tas': interpolated_data - 273.15
        })
        
        # 使用concat方法将新的数据与原有的DataFrame合并
        df = pd.concat([df, new_df], ignore_index=True)
    
    
    #%% 五天滑动平均求开始和结束
    date_objects = pd.to_datetime(df['datetime'])
    df.set_index(date_objects,inplace=True)
    df.drop(['datetime'], axis=1, inplace=True) 
    time_year=df.index.year.unique()
    
    # 创建保存array
    station_pairs = np.array([(name, id) for name, id in zip(station_name, station_id)]).ravel()
    station_pairs_list = station_pairs.tolist()
    column_names = ['年'] + station_pairs_list
    result_start_end = pd.DataFrame(columns=column_names)
    result_start_end_num = pd.DataFrame(columns=column_names)
    result_days = pd.DataFrame(columns=['年'] + station_id)
    result_hdd18 = pd.DataFrame(columns=['年'] + station_id)
    
    for idx,name in enumerate(station_id):
        n=0
        # break
        data=df[df['station_id_c']==name]
    
        tas_rolling = data['tas'].rolling(window=5)
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
        for year in time_year:
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
                result_hdd18.at[n,name]=(18-data_year.loc[first_index_z:last_index_z,'tas'].mean())*183
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

    #%% 计算信息加入
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



if __name__ == '__main__':
    
    #%% nc数据读取
    # NetCDF 文件路径列表
    tas_paths = [
        r'D:\Project\qh\Evaluate_Energy\data\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19500101-19501231.nc',
        r'D:\Project\qh\Evaluate_Energy\data\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19510101-19511231.nc',
        r'D:\Project\qh\Evaluate_Energy\data\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19520101-19521231.nc'
    ]

    # 插值站点信息
    station_id='52754,56151,52855,52862,56065,52645,56046,52955,52968,52963,52825,56067,52713,52943,52877,52633,52866'
    station_id = station_id.split(',')

    # 时间处理
    stats_times = '1950,1952'
    
    time_freq = 'Y'


    result_days,result_hdd18,result_start_end,result_start_end_num= winter_heating(tas_paths,station_id,time_freq,stats_times)
























