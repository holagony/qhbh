# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:49:08 2024

模式数据处理成站点数据
读取nc数据，插值到指定经纬度站点，使用邻近插值法

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

#cmip_df= cmip_data_deal(stats_path,sta_ids2,time_freq,stats_times,station_name,lon_list,lat_list,ele_dict[element])

# tas_paths=stats_path
# station_id=sta_ids2
# time_freq=time_freq
# stats_times=stats_times
# station_name=station_name
# target_lons=lon_list
# target_lats=lat_list
# ele_name=ele_dict[element]


def cmip_data_deal(tas_paths,station_id,time_freq,stats_times,station_name,target_lons,target_lats,ele_name):
    
    ele_name2=ele_name.split(',')

    # 模式数据对应的变量
    nc_dict=dict()
    nc_dict['TEM_Avg']='tas'
    nc_dict['TEM_Max']='tasmax'
    nc_dict['TEM_Min']='tasmin'
    nc_dict['PRE_Time_2020']='pr'
    nc_dict['win_s_2mi_avg']='ws'
    
    if len(ele_name2)==1:

        # 遍历所有文件
        df = pd.DataFrame()
    
        for tas_path in tas_paths:
            # print(tas_path)
            # break
            with nc.Dataset(tas_path) as tas_dataset:
                lon = tas_dataset.variables['lon'][:]
                lat = tas_dataset.variables['lat'][:]
                tas = tas_dataset.variables[nc_dict[ele_name2[0]]][:]
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
            
            stats_times_nc=stats_times.split(',')[0]+','+str(int(stats_times.split(',')[1])+1)
            date_indices = time_choose(time_freq,stats_times_nc,dates)
        
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
                ele_name2[0]: interpolated_data# - 273.15
            })
            
            # 使用concat方法将新的数据与原有的DataFrame合并
            df = pd.concat([df, new_df], ignore_index=True)
            
    elif len(ele_name2)==2:
        
        df = pd.DataFrame()
    
        for tas_path in tas_paths:
            with nc.Dataset(tas_path) as tas_dataset:
                lon = tas_dataset.variables['lon'][:]
                lat = tas_dataset.variables['lat'][:]
                tas = tas_dataset.variables[nc_dict[ele_name2[0]]][:]
                time = tas_dataset.variables['time'][:]
                time_var = tas_dataset.variables['time']
                time_calendar=time_var.calendar
                time_units = time_var.units
                ref_time_str = time_units.split('since ')[1]
            
            target_path = tas_path.replace(nc_dict[ele_name2[0]], nc_dict[ele_name2[1]])
            with nc.Dataset(target_path) as tas_dataset:
                tas2 = tas_dataset.variables[nc_dict[ele_name2[1]]][:]

            ref_date = datetime(int(ref_time_str[:4:]), int(ref_time_str[5:7:]),int(ref_time_str[8:10:]))
        
            lons, lats = np.meshgrid(lon, lat)
            
            if time_calendar=='365_day':
                dates = [increment_date(ref_date, int(t)) for t in time[:]]
            else:
                dates = [ref_date + timedelta(days=int(t)) for t in time[:]]
            
            stats_times_nc=stats_times.split(',')[0]+','+str(int(stats_times.split(',')[1])+1)
            date_indices = time_choose(time_freq,stats_times_nc,dates)
        
            tas = tas[date_indices,:, :]
            tas2 = tas2[date_indices,:, :]

            dates = [dates[i] for i in date_indices]
            
            points = np.vstack((lons.flatten(), lats.flatten())).T
            
            interpolated_data = np.zeros((len(dates) * len(target_lons),))
            interpolated_data2 = np.zeros((len(dates) * len(target_lons),))

            for i in range(len(dates)):
                values = tas[i, :, :].flatten()
                interpolated_data[i * len(target_lons):(i + 1) * len(target_lons)] = griddata(points, values, (target_lons, target_lats), method='nearest')
            
                values2 = tas2[i, :, :].flatten()
                interpolated_data2[i * len(target_lons):(i + 1) * len(target_lons)] = griddata(points, values2, (target_lons, target_lats), method='nearest')

            dates_repeated = np.repeat(dates, len(target_lons))
            
            station_id_repeated = np.tile(station_id, len(dates))
            
            # 创建一个新的DataFrame来存储当前循环的数据
            new_df = pd.DataFrame({
                'Datetime': dates_repeated,
                'Station_Id_C': station_id_repeated,
                'lon': np.tile(target_lons, len(dates)),
                'lat': np.tile(target_lats, len(dates)),
                ele_name2[0]: interpolated_data,
                ele_name2[1]: interpolated_data2# - 273.15

            })
            
            # 使用concat方法将新的数据与原有的DataFrame合并
            df = pd.concat([df, new_df], ignore_index=True)

    return df



