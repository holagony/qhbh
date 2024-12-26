# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:37:59 2024

@author: EDY
"""

import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import griddata
import os
from Utils.config import cfg
from scipy import spatial

def station_to_grid(lon_sta, lat_sta, value_sta, gridx, gridy, method, name):
    import numpy as np

    lon, lat = np.meshgrid(gridx, gridy)

    if method == 'kri':

        #%% 克里金

        #　variogram_model是变差函数模型，pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。
        # 使用不同的variogram_model，预测效果是不一样的，应该针对自己的任务选择合适的variogram_model。
        variogram_model = 'exponential'

        krig = OrdinaryKriging(lon_sta, lat_sta, value_sta, variogram_model=variogram_model, verbose=False, enable_plotting=False)
        data, ss3d = krig.execute("grid", gridx, gridy)

    elif method == 'ukri':

        #%% 泛克里金

        variogram_model = 'exponential'

        ukrig = UniversalKriging(lon_sta, lat_sta, value_sta, variogram_model=variogram_model, verbose=False, enable_plotting=False)
        data, ss3d = ukrig.execute("grid", gridx, gridy)

    elif method == 'idw3':

        #%% idw

        def haversine_dist(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
            radius = 6378.135E3  # radius of Earth, unit:m
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            arg = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            dist = 2 * radius * np.arcsin(np.sqrt(arg))
            return dist

        def interp_IDW(lon_sta, lat_sta, data_sta, lon2D, lat2D):
            n_sta = len(lon_sta)
            ny, nx = np.shape(lon2D)
            data2D = np.zeros((ny, nx))
            for j in range(ny):
                for i in range(nx):  #遍历二维每一个格点
                    dist = []  # 格点至所有站点的距离
                    for s in range(n_sta):
                        d = haversine_dist(lon_sta[s], lat_sta[s], lon2D[j, i], lat2D[j, i])
                        d = np.max([1.0, d])  # aviod divide by zero
                        dist.append(d)
                    wgt = 1.0 / np.power(dist, 2)
                    wgt_sum = np.sum(wgt)
                    arg_sum = np.sum(np.array(wgt) * np.array(data_sta))
                    data2D[j, i] = arg_sum / wgt_sum
            return data2D

        data = interp_IDW(lon_sta, lat_sta, value_sta, lon, lat)

        #%% windows临时方法，idw更快，呈现效果差
    elif method == 'idw2':
    
        def idw_interpolation(point_xy, point_z, grid_xy, k=12, p=2, offset=1e-10):
            # 计算球面距离
            def haversine_distance(p1, p2):
                lon1, lat1 = p1
                lon2, lat2 = p2
                R = 6371  # 地球平均半径(km)
                
                # 转换为弧度
                lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
                
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return R * c

            grid_shape = grid_xy[0].shape
            grid_flatten = np.reshape(grid_xy, (2, -1)).T
            
            # 使用KDTree找到最近邻点
            tree = spatial.cKDTree(point_xy)
            distances, idx = tree.query(grid_flatten, k=k)
            
            # 计算实际球面距离
            distances = np.array([[haversine_distance(grid_flatten[i], point_xy[j]) 
                                 for j in idx[i]] for i in range(len(grid_flatten))])
            
            distances = distances + offset  # 避免除零错误
            weights = 1.0 / (distances ** p)
            
            # 计算加权平均
            weighted_values = weights * point_z[idx]
            grid_z = np.sum(weighted_values, axis=1) / np.sum(weights, axis=1)
            
            return grid_z.reshape(grid_shape)

        # 准备输入数据
        point_xy = np.column_stack((lon_sta, lat_sta))
        point_z = value_sta
        grid_xy = (lon, lat)

        # 执行插值
        data = idw_interpolation(point_xy, point_z, grid_xy)
        data = data.astype(float)
        
    elif method == 'idw':
        
        import ctypes  

        def IDW_cppDll(x, y, z, xi, yi):
            # 加载动态库  
            if os.name == 'nt':
                idw_path=cfg.FILES.IDW_W
            elif os.name == 'posix':
                idw_path=cfg.FILES.IDW_L
            else:
                idw_path   =cfg.FILES.IDW_L

            lib = ctypes.CDLL(idw_path) 
            
            # 设置参数类型  
            lib.IDW_C_Interface.argtypes = [  
                ctypes.POINTER(ctypes.c_double),  # x  
                ctypes.POINTER(ctypes.c_double),  # y  
                ctypes.POINTER(ctypes.c_double),  # z  
                ctypes.c_size_t,                  # dataSize  
                ctypes.POINTER(ctypes.c_double),  # xi  
                ctypes.POINTER(ctypes.c_double),  # yi  
                ctypes.c_size_t,                  # interpSize  
                ctypes.POINTER(ctypes.c_double)   # results  
            ]  

            dataSize = len(z)
            interpSize = len(xi)
          
            x = (ctypes.c_double * dataSize)(*x)  
            y = (ctypes.c_double * dataSize)(*y)  
            z = (ctypes.c_double * dataSize)(*z)  
            xi = (ctypes.c_double * interpSize)(*xi)  
            yi = (ctypes.c_double * interpSize)(*yi)  
            results = (ctypes.c_double * (interpSize * 3))()  # 分配结果数组的空间  
            
            lib.IDW_C_Interface(x, y, z, dataSize, xi, yi, interpSize, results)  
            results = np.array(results).reshape(-1,3)
            return results
        
        wsp_idw = IDW_cppDll(lon_sta, lat_sta, value_sta, lon.flatten(), lat.flatten())
        wsp_idw = np.array(wsp_idw)
        data = np.reshape(wsp_idw[:,2], lon.shape) 
        
        
        #%% griddata
    elif method == 'griddata':
        data_points = np.column_stack([lon_sta, lat_sta])

        data_values = value_sta

        grid_points = np.column_stack([lon.ravel(), lat.ravel()])

        # 使用cubic方法进行插值
        result = griddata(data_points, data_values, grid_points, method='cubic')
        data = result.reshape(lon.shape)

    #%% 画图看结果自测使用
    '''
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib
    matplotlib.use('Agg')
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False 
    
    lon,lat=np.meshgrid(gridx,gridy)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    mesh = ax.contourf(lon,lat,data,transform=ccrs.PlateCarree())
    
    ax.scatter(lon_sta,lat_sta)
    
    for i in np.arange(len(lon_sta)):
        ax.text(lon_sta[i],lat_sta[i],str(value_sta[i]),fontsize=8)
    
    cbar = fig.colorbar(mesh, ax=ax, extend='both', shrink=0.8)#, ticks=np.linspace(levels[0],levels[-1],10))
    g1=ax.gridlines(draw_labels=True,linewidth=1,color='none',alpha=0.5,linestyle='--',x_inline=False,y_inline=False)#,xlocs=np.arange(105,125+5,5),ylocs=np.arange(32.5,45+2.5,2.5))
    g1.top_labels=False
    g1.right_labels=False
    g1.xformatter=LONGITUDE_FORMATTER
    g1.yformatter=LATITUDE_FORMATTER
    g1.rotate_labels=False
    
    result_picture = os.path.join(r'D:\Project\qh',name+'.png')
    fig.savefig(result_picture, dpi=200, bbox_inches='tight')
    plt.cla()
    '''

    return data


if __name__ == "__main__":

    #%% 数据读取与处理
    from Utils.data_processing import data_processing

    filename = r'D:\Project\3_项目\2_气候评估和气候可行性论证\qhkxxlz\Files\test_data\qh_year.csv'
    df = pd.read_csv(filename)
    yearly_df = data_processing(df)

    # 每个站求平均
    station_name = yearly_df['Station_Name'].unique()
    df_station = pd.DataFrame(columns=station_name)
    for name in station_name:
        df_station.at[0, name] = round(np.nanmean(yearly_df[yearly_df['Station_Name'] == name]['TEM_Max']), 2)
        df_station.at[1, name] = yearly_df[yearly_df['Station_Name'] == name]['Lon'].iloc[0]
        df_station.at[2, name] = yearly_df[yearly_df['Station_Name'] == name]['Lat'].iloc[0]

    df_station = df_station.T
    df_station.columns = ['TEM_Max', 'Lon', 'Lat']

    #%% 插值成网格（克里金、泛克里金、反距离加权）
    # 站点数据
    lon_sta = df_station['Lon']
    lat_sta = df_station['Lat']
    value_sta = df_station['TEM_Max']

    # 网格参数设置
    start_lon = 89.5
    start_lat = 31
    end_lon = 103.1
    end_lat = 39.3
    resolution = 0.01

    gridx = np.arange(start_lon, end_lon, resolution)
    gridy = np.arange(start_lat, end_lat, resolution)

    method = 'kri'
    data = station_to_grid(lon_sta, lat_sta, value_sta, gridx, gridy, method)
