import warnings
warnings.filterwarnings('ignore')

import os
import time
import glob
import psutil
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import geopandas as gpd
from datetime import datetime
from tqdm import tqdm
from Flood.wrapped.flood_model import SCS_CN, RFSM
from Utils.utils import tif_dataloader, array2nc, nc_save
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.config import cfg


def pre_simulation(param_A, param_b, param_C, param_n, r, p, t, total_t, lon, lat):
    '''
    利用暴雨强度公式芝加哥雨型，生成降水网格
    param_A/param_b/param_C/param_n 暴雨强度公式系数
    r 雨峰系数
    p 重现期
    t 降水时长 min
    total_t 模拟时长 h
    lon/lat 经纬度一维序列
    '''
    termination = int(t*r)
    pre_min = np.zeros(t)

    for k in range(t):
        i = k+1
        if i<= termination:
            tb = termination - i 
            pre_i = param_A*(1+param_C*np.log10(p))*((1-param_n)*tb/r + param_b) / (tb/r + param_b)**(1+param_n)
        else:
            ta = i - termination
            pre_i = param_A*(1+param_C*np.log10(p))*((1-param_n)*ta/(1-r) + param_b) / (ta/(1-r) + param_b)**(1+param_n)

        pre_min[k] = pre_i

    pre_hour = np.nansum(pre_min.reshape(int(t/60),60), axis=1) # shaple:(num_hours,1) 每小时的降水量
    sim_pre = np.zeros((total_t, lat.size, lon.size))

    for i in range(pre_hour.size):
        if i <= sim_pre.shape[0]:
            sim_pre[i] = sim_pre[i] + pre_hour[i]
        else:
            raise ValueError("Invalid location of simulated PRE")
            
    return sim_pre


class flood_model:
    '''
    太原市/山西省内涝模型
    '''

    def __init__(self, save_path, flag, pre_path, pre_type, previous, param_A, param_b, param_C, param_n, r, p, t, total_t):
        self.save_path = save_path
        self.flag = flag # TY/SHANXI
        self.pre_path = pre_path
        self.pre_type = pre_type
        self.previous = previous # 是否载入前一个时刻的CMPAS数据
        self.param_A = param_A
        self.param_b = param_b
        self.param_C = param_C
        self.param_n = param_n
        self.r = r
        self.p = p
        self.t = t
        self.total_t = total_t

    def get_gis(self):
        if self.flag == 'TY':
            dem_path = cfg.FILES.DEM_TY
            landuse_path = cfg.FILES.LANDUSE_TY
            watersh_path = cfg.FILES.WATERSH_TY
        else:
            dem_path = cfg.FILES.DEM_SHANXI
            landuse_path = cfg.FILES.LANDUSE_SHANXI
            watersh_path = cfg.FILES.WATERSH_SHANXI

        dem_data = tif_dataloader(dem_path, key='dem')
        dem = dem_data['data']
        dem[np.isnan(dem)] = 0
        lon = dem_data['lon_lat'][0][0, :] # 一维序列
        lat = dem_data['lon_lat'][1][:, 0] # 一维序列

        landuse_data = tif_dataloader(landuse_path, key='landuse')
        landuse = landuse_data['data']
        landuse[np.isnan(landuse)] = 0
        landuse[landuse == 99] = 12
        landuse[landuse == 0] = 12

        watersh_data = tif_dataloader(watersh_path, key='watersh')
        watersh = watersh_data['data']
        watersh[np.isnan(watersh)] = 0
        watersh = watersh + 1

        return dem, landuse, watersh, lon, lat

    def get_pre(self, lon, lat):
        '''
        获取和处理降水数据
        pre_type=1 MQPF文件
        pre_type=2 智能网格预报文件
        pre_type=4 根据雨型生成网格
        '''
        if self.pre_type == 1: # nc里面纬度从小到大
            ds = xr.open_dataset(self.pre_path)
            pre = ds.qpf_ml*5
            data_out = pre.interp(latitude=lat, longitude=lon, method=cfg.PARAMS.PRE_PROCESS_METHOD)
            data_out = data_out.data
            time_list = pd.to_datetime(ds.time)
            self.time = time_list[0]

        elif self.pre_type == 2: # nc里面纬度从小到大
            total_path = glob.glob(os.path.join(self.pre_path, '*.nc'))  # todo文件名排序
            ds_list = []
            time_list = []
            for num, path in enumerate(total_path):
                ds = xr.open_dataset(path)
                ds_list.append(ds)
                time = path.split('.')[0].split('_')[-1] # ER_2024082720_2024082800.nc
                st = datetime.strptime(time, '%Y%m%d%H')
                time_list.append(st)
                
                times = pd.date_range(st,st,freq='H')
                time_da = xr.DataArray(times, [('time', times)])
                ds = ds.expand_dims(time=time_da)

            ds_all = xr.concat(ds_list, dim='time')
            pre = ds_all.ER
            data_out = pre.interp(lat=lat, lon=lon, method=cfg.PARAMS.PRE_PROCESS_METHOD)
            data_out = data_out.data

        elif self.pre_type == 4: # 暴雨强度公式雨型
            data_out = pre_simulation(self.param_A, self.param_b, self.param_C, self.param_n, self.r, self.p, self.t, self.total_t, lon, lat)
            time = '2050010100' # 如果是仿真，起始时间固定2050-01-01 00时
            st = datetime.strptime(time, '%Y%m%d%H')
            et = st + dt.timedelta(hours=self.total_t - 1)
            time_list = pd.date_range(st, et, freq='1H')

        return data_out, time_list

    def cal_elevation(self, dem_data, landuse_data, watersh_data):
        '''
        整理生成各集水区高程序列和对应位置
        '''
        dem = dem_data
        landuse = landuse_data
        catchment = watersh_data

        dem[landuse == 51] += cfg.PARAMS.CITY_DSM_OFFSET
        dem[landuse == 52] += cfg.PARAMS.TOWM_DSM_OFFSET

        catchment_mark = np.unique(catchment)
        elevations = [dem[catchment == i + 1] for i in range(len(catchment_mark))]
        locations = [np.where(catchment == i + 1) for i in range(len(catchment_mark))]
        row_loc, clo_loc = zip(*locations)
        
        return elevations, row_loc, clo_loc

    def calc_flood(self):
        '''
        SCS+RFSM流程，保存nc
        pre_array: 载入的降水数据 3D array
        dem_data: 载入的DEM数据
        landuse_data: 载入的土地利用数据
        watersh_data: 载入的集水区数据
        '''
        dem_data, landuse_data, watersh_data, lon, lat = self.get_gis()
        pre_array, time_list = self.get_pre(lon, lat)
        elevations, row_loc, clo_loc = self.cal_elevation(dem_data, landuse_data, watersh_data)
        water_depth = np.zeros(pre_array.shape)

        # mqpf取前一个时刻的结果
        if self.pre_type == 1 and self.previous is not None:
            last_time = self.time - dt.timedelta(hours=1/6)
            last_time_str = last_time.strftime('%Y%m%d%H%M')
            previous_flood = os.path.join(self.previous, f'FLOOD_{self.flag}_' + str(self.pre_type) + '_' + last_time_str + '_024.nc')
            print('读取前一个时刻结果路径：' + previous_flood)

            if os.path.exists(previous_flood):
                previous_file = xr.open_dataset(previous_flood)
                last_water_depth = previous_file.FLOOD.data[0] # 第0个时刻的结果
                print('读取前一个时刻的结果成功')
            else:
                last_water_depth = np.zeros(pre_array[0].shape)
                print('读取失败，设定前一个时刻结果为空数组')
        else:
            last_water_depth = np.zeros(pre_array[0].shape)

        # 开始计算
        if self.pre_type == 1:
            fh = 1/6
        else:
            fh = 1

        for i in tqdm(range(pre_array.shape[0])):
            pre_temp = pre_array[i, :, :]
            
            # 针对山西智能网格数据 前24小时逐小时，后72小时逐3小时
            if (self.pre_type == 2) and (i > 23):
                fh = 3

            scs_model = SCS_CN(landuse_data, fh)
            runoff = scs_model.calc_runoff(pre_temp, last_water_depth)
            rfsm = RFSM(dem_data, landuse_data, watersh_data, runoff, elevations, row_loc, clo_loc)
            water_depth_now = rfsm.cal_water_depth()
            water_depth[i, :, :] = water_depth_now * 0.1
            last_water_depth = water_depth_now
            # print("max pre of {} is {:.2f}mm".format(i + 1, np.nanmax(pre_temp)))
            # print("max water_depth is {:.2f}cm".format(np.nanmax(water_depth_now) * 0.1))
            # print()

        result_dict = edict()
        water_depth = water_depth*cfg.PARAMS.DEPTH_SCALE

        # 先把原始数据保存为nc
        Data = {'PRE': pre_array, 'FLOOD': water_depth}
        ds = xr.Dataset()
        if cfg.INFO.SAVE_PRE:
            elements = ['PRE', 'FLOOD'] 
        else:
            elements = ['FLOOD']

        for var in elements:
            da = Data[var]
            ds = array2nc(da, lat, lon, var, time=time_list)

            if self.pre_type == 1:
                xarray_path = nc_save(ds, self.save_path, f"{var}_{self.flag}_{str(self.pre_type)}_{time_list[0].strftime('%Y%m%d%H%M')}_{'%03d'%len(time_list)}.nc")
            else:
                xarray_path = nc_save(ds, self.save_path, f"{var}_{self.flag}_{str(self.pre_type)}_{time_list[0].strftime('%Y%m%d%H')}_{'%03d'%len(time_list)}.nc")

            xarray_path = xarray_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)
            result_dict[var] = xarray_path

        # 然后根据路网的点插值
        if self.flag == 'TY':
            points = cfg.FILES.TY_ROAD_LEVEL1 # 1级道路
        else:
            points = cfg.FILES.SHANXI_ROAD_LEVEL1

        gdf = gpd.read_file(points)
        sta_list = gdf['useID'].tolist()
        lon_list = gdf['lon'].tolist()
        lat_list = gdf['lat'].tolist()
        interp_lon = xr.DataArray(lon_list, dims="location", coords={"location": sta_list,})
        interp_lat = xr.DataArray(lat_list, dims="location", coords={"location": sta_list,})
        selected_data = ds.interp(lat=interp_lat, lon=interp_lon, method='nearest')

        flood_tab = selected_data.FLOOD.data
        flood_tab = flood_tab.round(2)
        flood_tab = np.where(flood_tab>1, flood_tab, 0)
        flood_tab = pd.DataFrame(flood_tab, index=selected_data.time, columns=selected_data.location).T
        flood_tab.columns = [col.strftime('%Y%m%d%H%M') for col in flood_tab.columns]
        flood_tab = flood_tab.loc[~(flood_tab==0).all(axis=1)] # 删除全是0的行

        if self.pre_type == 1:
            csv_path = os.path.join(self.save_path,f"{var}_{self.flag}_{str(self.pre_type)}_{time_list[0].strftime('%Y%m%d%H%M')}_{'%03d'%len(time_list)}.csv")
        else:
            csv_path = os.path.join(self.save_path,f"{var}_{self.flag}_{str(self.pre_type)}_{time_list[0].strftime('%Y%m%d%H')}_{'%03d'%len(time_list)}.csv")

        flood_tab.to_csv(csv_path, encoding='utf-8')
        csv_path = csv_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)
        result_dict['FLOOD_CSV'] = csv_path

        return result_dict


if __name__ == '__main__':
    startime = datetime.now()
    save_file = r'C:\Users\MJY\Desktop\result'
    flag = 'TY' # TY or SHANXI
    pre_path = 'C:/Users/MJY/Desktop/shanxi_flood/zipdata/MQPF/mqpfshanxi_20240721_1650B.nc'
    pre_type = 1
    sf = flood_model(save_file, flag, pre_path, pre_type, previous=None, param_A=None, param_b=None, param_C=None, param_n=None, r=None, p=None, t=None, total_t=None)
    flood_ds = sf.calc_flood()
    endtime = datetime.now()
    runtime = endtime - startime
    print('spend time: %d seconds' % (runtime.seconds))
    print(u'memory_used: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("End : %s" % time.ctime())
    
