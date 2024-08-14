# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:22:08 2024

@author: EDY

分布图： 年平均气温、变率、最大值、最小值、距平、距平百分率、气候值、与上一年比较值、近10年均值、与近10年比较值
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from netCDF4 import Dataset
from Module01.wrapped.func01_table_stats import table_stats
from Utils.data_processing import data_processing
from Utils.station_to_grid import station_to_grid


def contour_picture(stats_result, data_df, shp_name, method, output_filepath):
    #%% 数据选取

    # 插值范围、掩膜
    gdf = gpd.read_file(shp_name)
    points = []
    for polygon in gdf.geometry:
        if polygon.geom_type == 'Polygon':
            exterior = polygon.exterior
            for point in exterior.coords:
                points.append(point)
        elif polygon.geom_type == 'MultiPolygon':
            for part in polygon:
                exterior = part.exterior
                for point in exterior.coords:
                    points.append(point)
    df_shp = pd.DataFrame(points, columns=['Longitude', 'Latitude'])

    # 站点经纬度匹配
    df_sta_1 = stats_result.T.reset_index()
    df_sta_1.columns = df_sta_1.iloc[0]
    df_sta_1 = df_sta_1.drop(df_sta_1.index[0])
    df_sta_1 = df_sta_1.iloc[:-5:, :]

    station_id = np.array(df_sta_1.iloc[:, 1])
    df_sta = pd.DataFrame(columns=['Station_id', 'lon', 'lat'])

    df_sta['Station_id'] = station_id
    for i in np.arange(len(station_id)):
        df_sta.loc[i, 'lon'] = float(data_df[data_df['Station_Id_C'] == station_id[i]]['Lon'].iloc[0])
        df_sta.loc[i, 'lat'] = float(data_df[data_df['Station_Id_C'] == station_id[i]]['Lat'].iloc[0])

    # 经纬度选取
    lon_sta = df_sta['lon'].values
    lat_sta = df_sta['lat'].values

    #%% 插值
    # 网格参数设置
    start_lon = df_shp['Longitude'].min() - 0.1
    start_lat = df_shp['Latitude'].min() - 0.1
    end_lon = df_shp['Longitude'].max() + 0.1
    end_lat = df_shp['Latitude'].max() + 0.1
    resolution = 0.1

    gridx = np.arange(start_lon, end_lon + resolution, resolution)
    gridy = np.arange(start_lat, end_lat + resolution, resolution)

    # result = dict()

    # 历年平均值
    df_sta_2 = df_sta_1.iloc[:, :-10:].T
    df_sta_2.columns = df_sta_2.iloc[0]
    df_sta_2 = df_sta_2.drop(df_sta_2.index[0])
    df_sta_2 = df_sta_2.drop(df_sta_2.index[0])

    df_sta_2.index = pd.DatetimeIndex(df_sta_2.index)
    df_sta_3 = df_sta_2.resample('Y').mean()
    year = df_sta_3.index.year

    output_filepath_name = os.path.join(output_filepath, 'data.nc')
    data = np.zeros((len(year), len(gridy), len(gridx)))

    for i in np.arange(len(year)):
        value_sta = df_sta_3.iloc[i, :].values

        # 数据清洗
        data_uclean = pd.DataFrame({'lon': lon_sta, 'lat': lat_sta, 'value': value_sta})

        # 将inf值替换为NaN
        data_uclean.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 移除包含NaN值的行
        data_clean = data_uclean.dropna()

        # 从清洗后的DataFrame中提取经度、纬度和值
        lon_clean = data_clean['lon'].values
        lat_clean = data_clean['lat'].values
        value_clean = data_clean['value'].values

        data[i, :, :] = station_to_grid(lon_clean, lat_clean, value_clean, gridx, gridy, method, str(year[i]))

    nc_file = nc.Dataset(output_filepath_name, 'w', format='NETCDF4', encoding='gbk')
    nc_file.createDimension('lon', gridx.shape[0])
    nc_file.createDimension('lat', gridy.shape[0])
    nc_file.createDimension('time', year.shape[0])

    lon_var = nc_file.createVariable('longitude', 'f4', ('lon', ))
    lon_var[:] = gridx

    lat_var = nc_file.createVariable('latitude', 'f8', ('lat', ))
    lat_var[:] = gridy

    time_var = nc_file.createVariable('time', 'f8', ('time', ))
    time_var[:] = year

    grid_var = nc_file.createVariable('data_year', 'f4', (
        'time',
        'lat',
        'lon',
    ))  # grid
    grid_var[:] = data

    nc_file.close()

    # 变率、最大值、最小值、距平、距平百分率、气候值、与上一年比较值、近10年均值、与近10年比较值
    ele_choose = ['变率', '最大值', '最小值', '距平', '距平百分率%', '参考时段均值', '与上一年比较值', '近10年均值', '与近10年比较值']

    i = 0
    for ele in ele_choose:
        value_sta = df_sta_1[ele].values
        ele_name = 'data' + str(i)

        # 数据清洗
        data_uclean = pd.DataFrame({'lon': lon_sta, 'lat': lat_sta, 'value': value_sta})

        # 将inf值替换为NaN
        data_uclean.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 移除包含NaN值的行
        data_clean = data_uclean.dropna()

        # 从清洗后的DataFrame中提取经度、纬度和值
        lon_clean = data_clean['lon'].values
        lat_clean = data_clean['lat'].values
        value_clean = data_clean['value'].values

        data2 = station_to_grid(lon_clean, lat_clean, value_clean, gridx, gridy, method, ele)

        nc_file = Dataset(output_filepath_name, 'a', format='NETCDF4', encoding='gbk')

        grid_var = nc_file.createVariable(ele_name, 'f4', (
            'lat',
            'lon',
        ))  # grid
        grid_var[:] = data2

        nc_file.close()
        i = i + 1

    # result['data'] = output_filepath_name

    return output_filepath_name, data, gridx, gridy, year


if __name__ == "__main__":

    path = r'D:\Project\3_项目\2_气候评估和气候可行性论证\qhkxxlz\Files\test_data\qh_mon.csv'

    df = pd.read_csv(path, low_memory=False)
    df = data_processing(df)
    data_df = df[df.index.year <= 5000]
    refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
    nearly_df = df[df.index.year > 2011]
    last_year = 2023
    time_freq = 'M1'
    ele = 'TEM_Avg'
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)

    output_filepath = r'D:\Project\1'
    shp_name = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\省界.shp'
    method = 'idw2'

    result, data, gridx, gridy, year = contour_picture(stats_result, data_df, shp_name, method, output_filepath)
