# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:53:10 2024

@author: EDY
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from netCDF4 import Dataset
from Utils.station_to_grid import station_to_grid
from Utils.config import cfg

# sta_ids=sta_ids2
# df=result_dfresult_df['表格']['历史']
# element='HD'
# shp_name=r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\省界.shp'

def contour_picture(df, shp_name,sta_ids, output_filepath,element):
    # 数据选取
    # 插值范围、掩膜
    gdf = gpd.read_file(shp_name)
    points = []
    for polygon in gdf.geometry:
        if polygon.geom_type == 'Polygon':
            exterior = polygon.exterior
            for point in exterior.coords:
                points.append(point)
        elif polygon.geom_type == 'MultiPolygon':
            for part in polygon.geoms:
                exterior = part.exterior
                for point in exterior.coords:
                    points.append(point)
    df_shp = pd.DataFrame(points, columns=['Longitude', 'Latitude'])
    
    # 网格参数设置
    start_lon = df_shp['Longitude'].min() - 0.1
    start_lat = df_shp['Latitude'].min() - 0.1
    end_lon = df_shp['Longitude'].max() + 0.1
    end_lat = df_shp['Latitude'].max() + 0.1
    resolution = 0.1

    gridx = np.arange(start_lon, end_lon + resolution, resolution)
    gridy = np.arange(start_lat, end_lat + resolution, resolution)
    
    
    # 匹配站点经纬度
    df_station=pd.read_csv(cfg.FILES.STATION,encoding='gbk')
    df_station['区站号']=df_station['区站号'].astype(str)
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': sta_ids}),df_station[['区站号', '站点名']],on='区站号')
    station_name = matched_stations['站点名'].values
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': sta_ids}),df_station[['区站号', '经度']],on='区站号')
    target_lons = matched_stations['经度'].values
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': sta_ids}),df_station[['区站号', '纬度']],on='区站号')
    target_lats = matched_stations['纬度'].values
    
    elem_dict=dict()
    elem_dict['HD']='采暖日'
    elem_dict['HDD18']='采暖度日'
    elem_dict['HDTIME']='HDTIME_NUM'

    # 历史数据
    data=df['表格']['历史'][elem_dict[element]]
    data=pd.DataFrame(data)
    data_year=data.iloc[:-4:,:-3:].copy()
    
    for i in np.arange(len(data_year)):
        value=data_year.iloc[i,1::]
        
        data_grid = station_to_grid(target_lons, target_lats, value, gridx, gridy, 'idw3', 'idw3')
