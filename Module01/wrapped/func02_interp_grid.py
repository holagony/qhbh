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
from Utils.config import cfg
from Module02.page_climate.wrapped.func_plot import polygon_to_mask


def contour_picture(stats_result, data_df, shp_name, method, output_filepath):
    # 数据选取
    # 插值范围、掩膜
    gdf = gpd.read_file(shp_name)
    # points = []
    # for polygon in gdf.geometry:
    #     if polygon.geom_type == 'Polygon':
    #         exterior = polygon.exterior
    #         for point in exterior.coords:
    #             points.append(point)
    #     elif polygon.geom_type == 'MultiPolygon':
    #         for part in polygon.geoms:
    #             exterior = part.exterior
    #             for point in exterior.coords:
    #                 points.append(point)
    # df_shp = pd.DataFrame(points, columns=['Longitude', 'Latitude'])


    # 站点经纬度匹配
    # df_sta_1 = stats_result.T.reset_index()
    # df_sta_1.columns = df_sta_1.iloc[0]
    # df_sta_1 = df_sta_1.drop(df_sta_1.index[0])
    df_sta_1 = stats_result.iloc[:,:-5:]

    station_id = np.array(df_sta_1.columns[1::])
    df_sta = pd.DataFrame(columns=['Station_id', 'lon', 'lat'])

    df_sta['Station_id'] = station_id
    for i in np.arange(len(station_id)):
        df_sta.loc[i, 'lon'] = float(data_df[data_df['Station_Id_C'] == station_id[i]]['Lon'].iloc[0])
        df_sta.loc[i, 'lat'] = float(data_df[data_df['Station_Id_C'] == station_id[i]]['Lat'].iloc[0])

    # 经纬度选取
    lon_sta = df_sta['lon'].values
    lat_sta = df_sta['lat'].values

    # 插值
    # 网格参数设置
    resolution = 0.5
    bounds = gdf['geometry'].total_bounds
    start_lon = bounds[0] - resolution
    start_lat = bounds[1] - resolution
    end_lon = bounds[2] + resolution
    end_lat = bounds[3] + resolution

    gridx = np.arange(start_lon, end_lon + resolution, resolution)
    gridy = np.arange(start_lat, end_lat + resolution, resolution)

    # result = dict()

    # 历年平均值
    df_sta_2 = df_sta_1.iloc[:-10:,:]
    # df_sta_2.columns = df_sta_2.iloc[0]
    # df_sta_2 = df_sta_2.drop(df_sta_2.index[0])
    # df_sta_2 = df_sta_2.drop(df_sta_2.index[0])
    df_sta_2['时间'] = pd.to_datetime(df_sta_2['时间'], format='%Y')
    df_sta_2.index = pd.DatetimeIndex(df_sta_2['时间'])
    df_sta_2.drop(['时间'], axis=1, inplace=True) 

    df_sta_3 = df_sta_2.resample('Y').mean()
    year = df_sta_3.index.year

    output_filepath_name = os.path.join(output_filepath, 'grid_data.nc')
    data = np.zeros((len(year), len(gridy), len(gridx)))

    year_u = []
    for i in np.arange(len(year)):
        try:
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

            if len(value_clean) == 0:
                continue

            year_u.append(year[i])
            grid = station_to_grid(lon_clean, lat_clean, value_clean, gridx, gridy, method, str(year[i]))

            # 新增掩膜
            # multi_polygon = gdf['geometry'].unary_union
            # lon_grid, lat_grid = np.meshgrid(gridx, gridy)
            # mask = polygon_to_mask(multi_polygon, lon_grid, lat_grid)
            # mask = np.where(mask == False, 1, 0)  # 生成mask，并将True/False转化为0/1
            # mask_grid = np.ma.masked_array(grid, mask, fill_value=np.nan)
            # mask_grid = mask_grid.filled()
            # data[i, :, :] = mask_grid
            data[i, :, :] = grid
        except:
            data[i, :, :] = np.nan

    year_u = np.array(year_u)
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
    ele_choose = ['变率', '最大值', '最小值', '距平', '距平百分率', '参考时段均值', '与上一年比较值', '近10年均值', '与近10年比较值']

    df_sta_1=df_sta_1.T
    df_sta_1.columns = df_sta_1.iloc[0]
    df_sta_1 = df_sta_1.drop(df_sta_1.index[0])
    
    i = 0
    for ele in ele_choose:
        try:
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

            if len(value_clean) == 0:
                i = i + 1
                continue

            data2 = station_to_grid(lon_clean, lat_clean, value_clean, gridx, gridy, method, ele)

            # 新增掩膜
            # multi_polygon = gdf['geometry'].unary_union
            # lon_grid, lat_grid = np.meshgrid(gridx, gridy)
            # mask = polygon_to_mask(multi_polygon, lon_grid, lat_grid)
            # mask = np.where(mask == False, 1, 0)  # 生成mask，并将True/False转化为0/1
            # mask_grid = np.ma.masked_array(data2, mask, fill_value=np.nan)
            # mask_grid = mask_grid.filled()
            # data[i, :, :] = mask_grid

        except:
            data2 = np.nan

        nc_file = Dataset(output_filepath_name, 'a', format='NETCDF4', encoding='gbk')

        grid_var = nc_file.createVariable(ele_name, 'f4', ('lat','lon',))  # grid
        grid_var[:] = data2

        nc_file.close()
        i = i + 1

    # result['data'] = output_filepath_name
    return output_filepath_name, data, gridx, gridy, year


if __name__ == "__main__":
    path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_year.csv'
    df = pd.read_csv(path, low_memory=False)
    element = 'TEM_Avg'
    df = df[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Datetime', 'Year', element]]
    df = data_processing(df, element)
    
    data_df = df[(df.index.year >= 1981) & (df.index.year <= 2023)]
    refer_df = df[(df.index.year >= 1991) & (df.index.year <= 2020)]
    nearly_df = df[(df.index.year >= 2014) & (df.index.year <= 2023)]
    last_year = 2023
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, element, last_year)

    output_filepath = r'C:/Users/MJY/Desktop/result'
    shp_name = r'C:\Users\MJY\Desktop\qhbh\zipdata\shp\qh\qh.shp'
    method = 'ukri'
    result, data, gridx, gridy, year = contour_picture(stats_result, data_df, shp_name, method, output_filepath)
