# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:15:37 2024

@author: EDY
"""


import netCDF4 as nc
from datetime import  date,datetime, timedelta
import numpy as np
import pandas as pd
from Utils.config import cfg
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# 求得区域平均，区域指的是传进来的所有站点
def data_proce(df,processing_methods, additional_method=None):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['年份'] = df['Datetime'].dt.year
    
    if additional_method:
        processing_methods = {col: additional_method for col in processing_methods.keys()}
    else:
        processing_methods = processing_methods
        
    # 动态处理每个分组
    def process_group(group):
        processed = {
            'Station_Id_C': group['Station_Id_C'].iloc[0],
            'Datetime': group['Datetime'].iloc[0]
        }
        for element, method in processing_methods.items():
            if element in group.columns:
                processed[f'{element}_{method}'] = getattr(group[element], method)()
        return pd.Series(processed)
    
    grouped_data = df.groupby(['Station_Id_C', '年份']).apply(process_group).reset_index(drop=True)
    
    # 保存站点数据
    station_data=grouped_data.copy()
    station_data['年'] = station_data['Datetime'].dt.year
    station_data.drop(['Datetime'], axis=1, inplace=True) 

    
    grouped_data['年份'] = grouped_data['Datetime'].dt.year
    average_data = grouped_data.iloc[:,2::].groupby('年份').mean()
    
    return station_data,average_data

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


def time_choose(time_freq,time_freq_data,stats_times,dates):        


    if time_freq== 'Y':
        # Y
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        date_indices = [i for i, date in enumerate(dates) if int(start_year) <= date.year <= int(end_year)]
    
    elif time_freq== 'Q':

        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        month = time_freq_data
        month = list(map(int,month.split(',')))
        
        if 12 in  month:
            date_indices = [
                i for i, date in enumerate(dates) 
                if ((start_year <= date.year <= end_year) & (date.month in month))
                if (date.year == start_year - 1) & (date.month == 12)
                if (date.year == end_year + 1) & (date.month in [1, 2])
            ] 
        else:
            date_indices = [i for i, date in enumerate(dates) if ((int(start_year) <= date.year <= int(end_year)) & (date.month in month))]

    elif time_freq== 'M2':
    
        # M2
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        month = time_freq_data
        month = list(map(int,month.split(',')))
        
        date_indices = [i for i, date in enumerate(dates) if ((int(start_year) <= date.year <= int(end_year)) & (date.month in month))]
    elif time_freq== 'D1':
    
        # D1
        start_time = stats_times.split(',')[0]+'0101'
        end_time = stats_times.split(',')[1]+'1231'
        
        start_date_nc_object = datetime.strptime(start_time, '%Y%m%d')
        end_date_nc_object = datetime.strptime(end_time, '%Y%m%d')
        date_indices = [i for i, date in enumerate(dates) if start_date_nc_object <= date <= end_date_nc_object]
    
    return date_indices


def model_factor_data_deal(tas_paths,station_id,var,ele,time_freq,time_freq_data,stats_times,processing_methods):
    
    #%% 站点编号 和 站点名经纬度匹配
    df_station=pd.read_csv(cfg.FILES.STATION,encoding='gbk')
    df_station['区站号']=df_station['区站号'].astype(str)
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '站点名']],on='区站号')
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '经度']],on='区站号')
    target_lons = matched_stations['经度'].values
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '纬度']],on='区站号')
    target_lats = matched_stations['纬度'].values
    
    # 初始化 DataFrame
    df = pd.DataFrame(columns=['Datetime', 'Station_Id_C', 'lon', 'lat', ele])
    
    # 参考日期
    
    # 遍历所有文件
    for tas_path in tas_paths:
        # break
        with nc.Dataset(tas_path) as tas_dataset:
            lon = tas_dataset.variables['lon'][:]
            lat = tas_dataset.variables['lat'][:]
            tas = tas_dataset.variables[var][:]
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
        
        
        date_indices = time_choose(time_freq,time_freq_data,stats_times,dates)
    
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
            'Datetime': dates_repeated,
            'Station_Id_C': station_id_repeated,
            'lon': np.tile(target_lons, len(dates)),
            'lat': np.tile(target_lats, len(dates)),
            ele: interpolated_data
        })
        
        # 使用concat方法将新的数据与原有的DataFrame合并
        df = pd.concat([df, new_df], ignore_index=True)
    
    # 针对不同的要素进行不同的要素处理

    if time_freq != 'D':
        verify_station_data,verify_data_deal=data_proce(df,processing_methods)
    else:
        verify_station_data=df.copy()
        verify_data=df.copy()
        verify_data['Datetime'] = pd.to_datetime(verify_data['Datetime'])
        verify_data=verify_data.set_index(verify_data['Datetime'])
        verify_data.drop([ 'Station_Id_C','Datetime'], axis=1,inplace=True)
        verify_data_deal = verify_data.resample('D').mean()
    
    
    return verify_station_data,verify_data_deal



if __name__ == '__main__':
    
    tas_paths = [
        r'D:\Project\qh\Evaluate_Energy\data\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19500101-19501231.nc',
        r'D:\Project\qh\Evaluate_Energy\data\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19510101-19511231.nc',
        r'D:\Project\qh\Evaluate_Energy\data\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19520101-19521231.nc'
    ]
    
    # 插值站点信息
    sta_ids='51886,52737,52876'
    station_id = sta_ids.split(',')
    var='tas'
    ele='TEM_Avg'
    # 时间处理
    time_freq='Y'
    time_freq_data='0'    
    stats_times='1950,1952'
    
    resample_max = ['TEM_Max', 'PRS_Max', 'WIN_S_Max', 'WIN_S_Inst_Max', 'GST_Max', 'huangku']

    resample_min = ['TEM_Min', 'PRS_Min', 'GST_Min', 'RHU_Min', 'fanqing']

    resample_sum = ['SSH','PRE_Time_2020', 'PRE_Days', 'EVP_Big', 'EVP', 'EVP_Taka', 'PMET','sa','rainstorm','light_snow','snow',
                    'medium_snow','heavy_snow','severe_snow','Hail_Days','Hail','GaWIN',
                    'GaWIN_Days','SaSt','SaSt_Days','FlSa','FlSa_Days','FlDu','FlDu_Days',
                    'Thund','Thund_Days''high_tem','drought','light_drought','medium_drought',
                    'heavy_drought','severe_drought','Accum_Tem']

    resample_mean = ['TEM_Avg', 'PRS_Avg', 'WIN_S_2mi_Avg', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Avg_5cm', 'GST_Avg_10cm', 
                     'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 
                     'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 'SSH', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg', 'Cov', 'pmet']

    processing_methods = {element: 'mean' for element in resample_mean}
    processing_methods.update({element: 'sum' for element in resample_sum})
    processing_methods.update({element: 'max' for element in resample_max})
    processing_methods.update({element: 'min' for element in resample_min})
    
    verify_station_data,verify_data_deal=model_factor_data_deal(tas_paths,station_id,var,ele,time_freq,time_freq_data,stats_times,processing_methods)
























