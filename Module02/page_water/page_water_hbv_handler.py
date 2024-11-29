import os
import uuid
import numpy as np
import pandas as pd
import xarray as xr
import psycopg2
from io import StringIO
from psycopg2 import sql
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.data_processing import data_processing
import time
import netCDF4 as nc
from datetime import  date,datetime, timedelta
from Module02.page_water.wrapped.hbv import hbv_main
from Module02.page_water.wrapped.func01_q_stats import stats_q
from Module02.page_water.wrapped.func02_result_stats import stats_result_1, stats_result_2, stats_result_3
import glob
from Utils.read_model_data import read_model_data


# hbv计算接口


def choose_mod_path(inpath, data_source, insti, var, time_scale, yr, expri_i, res=None):
    # cmip数据路径选择
    """
    :param inpath: 根路径目录
    :param data_source: 数据源 original/Delat/PDF/RF
    :param insti: 数据机构 BCC-CSM2-MR...
    :param var: 要素缩写 气温tas 降水pr-new
    :param time_scale: 数据时间尺度 daily
    :param yr: 年份
    :param expri_i: 场景
    :param res: 分辨率
    :return: 数据所在路径、文件名
    """
    if yr < 2015:
        expri = 'historical'
    else:
        expri = expri_i

    if time_scale == 'daily':
        path1 = 'daily'
        # filen = var + '_day_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    elif time_scale == 'monthly':
        path1 = 'monthly'
        # filen = var + '_month_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    elif time_scale == 'yearly':
        path1 = 'yearly'
        # filen = var + '_year_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    else:
        path1 = time_scale
        # filen = var + '_' + time_scale + '_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'

    if data_source=='original':
        # path = os.path.join(inpath, data_cource,path1,insti ,expri,var,filen)
        path_dir = os.path.join(inpath, data_source,path1,insti ,expri,var)
        path=glob.glob(os.path.join(path_dir, f'{var}*{str(yr)}0101*.nc'))[0]
    else:
        # path = os.path.join(inpath, data_cource,res,path1,insti ,expri,var,filen)
        path_dir = os.path.join(inpath, data_source,res,path1,insti ,expri,var)
        path=glob.glob(os.path.join(path_dir, f'{var}*{str(yr)}0101*.nc'))[0]

    return path

# inpath = r'C:\Users\MJY\Desktop\qhbh\zipdata\cmip6' # cmip6路径
# data_source = 'original'
# insti = 'BCC-CSM2-MR'
# var = 'pr'
# time_scale = 'daily'
# expri_i = 'ssp126'
# yr = 2018
# path = choose_mod_path(inpath, data_source, insti, var, time_scale, yr, expri_i, res=None)

def hbv_single_calc(data_json):
    '''
    获取天擎数据，参数说明
    :param refer_years: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'

    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'
        季 - 'Q'
        月(连续) - 'M1'
        月(区间) - 'M2' 
        日(连续) - 'D1'
        日(区间) - 'D2'

    :param evaluate_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'
        (2)当time_freq选择季Q。下载连续的月数据，处理成季数据（多下两个月数据），提取历年相应的季节数据，传：['%Y,%Y','3,4,5']，其中'3,4,5'为季度对应的月份 
        (3)当time_freq选择月(连续)M1。下载连续的月数据，传参：'%Y%m,%Y%m'
        (4)当time_freq选择月(区间)M2。下载连续的月数据，随后提取历年相应的月数据，传参：['%Y,%Y','11,12,1,2'] 前者年份，'11,12,1,2'为连续月份区间
        (5)当time_freq选择日(连续)D1。下载连续的日数据，传参：'%Y%m%d,%Y%m%d'
        (6)当time_freq选择日(区间)D2。直接调天擎接口，下载历年区间时间段内的日数据，传：['%Y,%Y','%m%d,%m%d'] 前者年份，后者区间
    
    :param hydro_ids: 传入的水文站点

    :param sta_ids: 传入的期限站点

    '''
    # 1.参数读取
    time_freq = data_json['time_freq'] # 控制预估时段
    evaluate_times = data_json['evaluate_times'] # 预估时段时间条
    refer_years = data_json['refer_years'] # 参考时段时间条
    valid_times = data_json['valid_times'] # 验证期 '%Y%m,%Y%m'
    hydro_ids = data_json['hydro_ids'] # 水文站 40100350 唐乃亥
    sta_ids = data_json['sta_ids'] # 水文站对应的气象站 唐乃亥对应 '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    cmip_type = data_json['cmip_type'] # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    cmip_res = data_json.get('cmip_res') # 分辨率 1/5/10/25/50/100 km
    cmip_model = data_json['cmip_model'] # 模式，列表：['CanESM5','CESM2']等
    d = data_json['d']
    fc = data_json['fc']
    beta = data_json['beta']
    c = data_json['c']
    k0 = data_json['k0']
    k1 = data_json['k1']
    k2 = data_json['k2']
    kp = data_json['kp']
    pwp = data_json['pwp']
    Tsnow_thresh = data_json['Tsnow_thresh']
    ca = data_json['ca']
    l = data_json['l']
    
    # 2.参数处理
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)
    
    if '集合' in cmip_model:
        cmip_model.remove('集合')
        cmip_model.append('Set')
        
    ######################################################
    # 数据下载
    ##### 下载验证期时段 & 参考时段的水文数据，展示 （对应表格-观测）
    hydro_ids = tuple(hydro_ids.split(','))
    elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,adnm,Q'
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
    query = sql.SQL(f"""
                    SELECT {elements}
                    FROM public.qh_climate_other_river_day
                    WHERE
                        ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s)
                        OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                        OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s))
                        AND station_id_c IN %s
                    """)

    start_year = valid_times.split(',')[0][:4]
    end_year = valid_times.split(',')[1][:4]
    start_month = valid_times.split(',')[0][4:]
    end_month = valid_times.split(',')[1][4:]
    cur.execute(query, (start_year, start_month, start_year, end_year, end_year, end_month, hydro_ids))
    data = cur.fetchall()
    data_df = pd.DataFrame(data)
    data_df.columns = elements.split(',')

    # 参考时段的水文站数据 用于计算距平
    query = sql.SQL(f"""
                    SELECT {elements}
                    FROM public.qh_climate_other_river_day
                    WHERE
                        CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                        AND station_id_c IN %s
                    """)

    start_year = refer_years.split(',')[0]
    end_year = refer_years.split(',')[1]
    cur.execute(query, (start_year, end_year, hydro_ids))
    data = cur.fetchall()
    refer_df = pd.DataFrame(data)
    refer_df.columns = elements.split(',')

    ##### 下载验证期时段相应的气象数据并处理，用于HBV计算 （对应表格-模拟（观测））
    sta_ids = tuple(sta_ids.split(','))
    elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,TEM_Avg,PRE_Time_2020'
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
    query = sql.SQL(f"""
                    SELECT {elements}
                    FROM public.qh_qhbh_cmadaas_day
                    WHERE
                        ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s)
                        OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                        OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s))
                        AND station_id_c IN %s
                    """)

    start_year = valid_times.split(',')[0][:4]
    end_year = valid_times.split(',')[1][:4]
    start_month = valid_times.split(',')[0][4:]
    end_month = valid_times.split(',')[1][4:]
    cur.execute(query, (start_year, start_month, start_year, end_year, end_year, end_month, sta_ids))
    data = cur.fetchall()
    data_df_meteo = pd.DataFrame(data)
    data_df_meteo.columns = elements.split(',')

    # 关闭数据库
    cur.close()
    conn.close()
    
    # 读取数据
    res_d = dict()
    res_d['25'] = '0.25deg'
    res_d['50'] = '0.52deg'
    res_d['100'] = '1deg'
    
    if os.name == 'nt':
        data_dir = r'C:\Users\MJY\Desktop\excel_data\csv' # 本地
    else:
        if cmip_type == 'original':
            data_dir = '/model_data/station_data/csv' # 容器内
        elif cmip_type == 'delta':
            data_dir = '/model_data/station_data_delta/csv' # 容器内
            data_dir = os.path.join(data_dir, res_d[cmip_res])
                
    ##### 下载验证期时段的cmip6数据，并插值到站点，用于HBV计算，蒸发数据来自气象站（对应表格-模拟（模式）） 
    start_year = int(valid_times.split(',')[0][:4])
    end_year = int(valid_times.split(',')[1][:4])
    
    # 读取数据
    # 原始的读取nc
    # inpath = r'C:\Users\MJY\Desktop\qhbh\zipdata\cmip6'
    # vaild_cmip = dict()
    # for exp in ['ssp126','ssp245','ssp585']:
    #     vaild_cmip[exp] = dict()
    #     for insti in cmip_model:
    #         vaild_cmip[exp][insti] = dict()
    #         tmp_lst = []
    #         pre_lst = []
    #         for year in range(start_year,end_year+1):
    #             tem_file_path = choose_mod_path(inpath=inpath, 
    #                                             data_source=cmip_type, 
    #                                             insti=insti, 
    #                                             var='tas', 
    #                                             time_scale='daily', 
    #                                             yr=year, 
    #                                             expri_i=exp, 
    #                                             res=cmip_res)

    #             pre_file_path = choose_mod_path(inpath=inpath, 
    #                                             data_source=cmip_type, 
    #                                             insti=insti, 
    #                                             var='pr', 
    #                                             time_scale='daily', 
    #                                             yr=year, 
    #                                             expri_i=exp, 
    #                                             res=cmip_res)
                
    #             ds_tmp = xr.open_dataset(tem_file_path)
    #             pre_tmp = xr.open_dataset(pre_file_path)
    #             tmp_lst.append(ds_tmp)
    #             pre_lst.append(pre_tmp)
            
    #         tmp_all = xr.concat(tmp_lst, dim='time')
    #         try:
    #             tmp_all['time'] = tmp_all.indexes['time'].to_datetimeindex().normalize()
    #         except:
    #             tmp_all['time'] = tmp_all.indexes['time'].normalize()
    #         pre_all = xr.concat(pre_lst, dim='time')
            
    #         try:
    #             pre_all['time'] = pre_all.indexes['time'].to_datetimeindex().normalize()
    #         except:
    #             pre_all['time'] = pre_all.indexes['time'].normalize()
    #         vaild_cmip[exp][insti]['tmp'] = tmp_all
    #         vaild_cmip[exp][insti]['pre'] = pre_all
    
    # 直接读取csv
    vaild_cmip = dict()
    station_id = list(sta_ids)
    time_scale= 'daily'
    time_freq_tmp = 'M1' # 验证期格式固定对应M1
    for exp in ['ssp126', 'ssp245', 'ssp585']:
        vaild_cmip[exp] = dict()
        for insti in cmip_model:
            vaild_cmip[exp][insti] = dict()
            
            # 读取tem & 合成nc
            excel_data_tas = read_model_data(data_dir,time_scale,insti,exp,'tas',valid_times,time_freq_tmp,station_id)
            time_tmp = excel_data_tas.index
            excel_data_tas = excel_data_tas.mean(axis=1) # 多站求平均，代表水文站
            # location_tmp = excel_data_tas.columns.tolist()
            # da = xr.DataArray(excel_data_tas.values, coords=[time_tmp, location_tmp], dims=['time', 'location'])
            da = xr.DataArray(excel_data_tas.values, coords=[time_tmp], dims=['time'])
            ds = xr.Dataset({'tas': da.astype('float32')})
            vaild_cmip[exp][insti]['tmp'] = ds
            
            # 读取pre & 合成nc
            excel_data_pr = read_model_data(data_dir,time_scale,insti,exp,'pr',valid_times,time_freq_tmp,station_id)
            excel_data_pr = excel_data_pr.mean(axis=1) # 多站求平均，代表水文站
            time_tmp = excel_data_pr.index
            
            # location_tmp = excel_data_pr.columns.tolist()
            da = xr.DataArray(excel_data_pr.values, coords=[time_tmp], dims=['time'])
            ds = xr.Dataset({'pr': da.astype('float32')})
            vaild_cmip[exp][insti]['pre'] = ds
            
    ##### 下载预估时段的cmip6数据，并插值到站点，用于HBV计算，使用预估时间（在这个里面生成预估气象数据，对应预估）
    if time_freq == 'Y':
        start_year = int(evaluate_times.split(',')[0])
        end_year = int(evaluate_times.split(',')[1])
    elif time_freq in ['Q', 'M2', 'D2']:
        start_year = int(evaluate_times[0].split(',')[0])
        end_year = int(evaluate_times[0].split(',')[1])
    elif time_freq in ['M1', 'D1']:
        start_year = int(evaluate_times.split(',')[0][:4])
        end_year = int(evaluate_times.split(',')[1][:4])

    # evaluate_cmip = dict()
    # for exp in ['ssp126','ssp245','ssp585']:
    #     evaluate_cmip[exp] = dict()
    #     for insti in cmip_model:
    #         evaluate_cmip[exp][insti] = dict()
    #         tmp_lst = []
    #         pre_lst = []
    #         for year in range(start_year,end_year+1):
    #             tem_file_path = choose_mod_path(inpath=inpath, 
    #                                             data_source=cmip_type, 
    #                                             insti=insti, 
    #                                             var='tas', 
    #                                             time_scale='daily', 
    #                                             yr=year, 
    #                                             expri_i=exp, 
    #                                             res=cmip_res)

    #             pre_file_path = choose_mod_path(inpath=inpath, 
    #                                             data_source=cmip_type, 
    #                                             insti=insti, 
    #                                             var='pr', 
    #                                             time_scale='daily', 
    #                                             yr=year, 
    #                                             expri_i=exp, 
    #                                             res=cmip_res)
                
    #             ds_tmp = xr.open_dataset(tem_file_path)
    #             pre_tmp = xr.open_dataset(pre_file_path)
    #             tmp_lst.append(ds_tmp)
    #             pre_lst.append(pre_tmp)
            
    #         tmp_all = xr.concat(tmp_lst, dim='time')
    #         try:
    #             tmp_all['time'] = tmp_all.indexes['time'].to_datetimeindex().normalize()
    #         except:
    #             tmp_all['time'] = tmp_all.indexes['time'].normalize()

    #         pre_all = xr.concat(pre_lst, dim='time')
    #         try:
    #             pre_all['time'] = pre_all.indexes['time'].to_datetimeindex().normalize()
    #         except:
    #             pre_all['time'] = pre_all.indexes['time'].normalize()
                
    #         evaluate_cmip[exp][insti]['tmp'] = tmp_all
    #         evaluate_cmip[exp][insti]['pre'] = pre_all

    # 直接读取csv
    evaluate_cmip = dict()
    station_id = list(sta_ids)
    time_scale= 'daily'
    for exp in ['ssp126', 'ssp245', 'ssp585']:
        evaluate_cmip[exp] = dict()
        for insti in cmip_model:
            evaluate_cmip[exp][insti] = dict()
            
            # 读取tem & 合成nc
            excel_data_tas = read_model_data(data_dir,time_scale,insti,exp,'tas',evaluate_times,time_freq,station_id)
            excel_data_tas = excel_data_tas.mean(axis=1) # 多个站点取平均，计算结果代表水文站
            time_tmp = excel_data_tas.index
            da = xr.DataArray(excel_data_tas.values, coords=[time_tmp], dims=['time'])
            ds = xr.Dataset({'tas': da.astype('float32')})
            evaluate_cmip[exp][insti]['tmp'] = ds
            
            # 读取pre & 合成nc
            excel_data_pr = read_model_data(data_dir,time_scale,insti,exp,'pr',evaluate_times,time_freq,station_id)
            excel_data_pr = excel_data_pr.mean(axis=1) # 多个站点取平均，计算结果代表水文站
            time_tmp = excel_data_pr.index
            da = xr.DataArray(excel_data_pr.values, coords=[time_tmp], dims=['time'])
            ds = xr.Dataset({'pr': da.astype('float32')})
            evaluate_cmip[exp][insti]['pre'] = ds

    ######################################################
    # 数据处理
    ##### 水文数据处理，日尺度
    data_df['Datetime'] = pd.to_datetime(data_df['Datetime'],format='%Y-%m-%d')
    data_df.set_index('Datetime', inplace=True)
    data_df.sort_index(inplace=True)
    data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    data_df['Lon'] = data_df['Lon'].astype(float)
    data_df['Lat'] = data_df['Lat'].astype(float)
    data_df['Q'] = data_df['Q'].apply(lambda x: float(x) if x != '' else np.nan)

    refer_df['Datetime'] = pd.to_datetime(refer_df['Datetime'],format='%Y-%m-%d')
    refer_df.set_index('Datetime', inplace=True)
    refer_df.sort_index(inplace=True)
    refer_df['Station_Id_C'] = refer_df['Station_Id_C'].astype(str)
    refer_df['Lon'] = refer_df['Lon'].astype(float)
    refer_df['Lat'] = refer_df['Lat'].astype(float)
    refer_df['Q'] = refer_df['Q'].apply(lambda x: float(x) if x != '' else np.nan)#.astype(float) # 日尺度

    ##### 验证期的气象数据,日尺度，后续区域平均
    data_df_meteo['Datetime'] = pd.to_datetime(data_df_meteo['Datetime'])
    data_df_meteo.set_index('Datetime', inplace=True)
    data_df_meteo.sort_index(inplace=True)
    data_df_meteo['Station_Id_C'] = data_df_meteo['Station_Id_C'].astype(str)
    data_df_meteo['Lon'] = data_df_meteo['Lon'].astype(float)
    data_df_meteo['Lat'] = data_df_meteo['Lat'].astype(float)
    
    # TODO 未来应该改成用月平均气温和降水计算
    data_df_meteo['EVP_Taka'] = 3100*data_df_meteo['TEM_Avg']/(3100+1.8*(data_df_meteo['PRE_Time_2020']**2)*np.exp((-34.4*data_df_meteo['TEM_Avg'])/(235+data_df_meteo['TEM_Avg'])))

    ##### 验证期的cmip6插值到气象站，然后平均，作为对应的水文站
    # 首先筛选时间
    # valid_times格式: "%Y%m,%Y%m" '200502,201505'
    s = valid_times.split(',')[0]
    e = valid_times.split(',')[1]
    s = pd.to_datetime(s,format='%Y%m')
    e = pd.to_datetime(e,format='%Y%m') + pd.DateOffset(months=1)
    time_index = pd.date_range(start=s, end=e, freq='D')[:-1] # M1
    time_index = time_index[~((time_index.month==2) & (time_index.day==29))] # 由于数据原因，删除2.29

    hydro_lon = data_df['Lon'][0]
    hydro_lat = data_df['Lat'][0]
    for _, sub_dict1 in vaild_cmip.items():  # vaild_cmip[exp][insti]['tmp']
        for _, sub_dict2 in sub_dict1.items():
            for key, ds_data in sub_dict2.items():
                try:
                    selected_data = ds_data.sel(time=time_index)
                except:
                    selected_data = ds_data
                # selected_data = selected_data.interp(lat=hydro_lat, lon=hydro_lon, method='nearest')
                sub_dict2[key] = selected_data

    ##### 预估期的cmip6插值到气象站，然后平均，作为对应的水文站
    # 首先筛选时间
    if time_freq == 'Y':
        s = evaluate_times.split(',')[0]
        e = evaluate_times.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1] # 'Y'

    elif time_freq in ['Q', 'M2']:
        s = evaluate_times[0].split(',')[0]
        e = evaluate_times[0].split(',')[1]
        mon_list = [int(val) for val in evaluate_times[1].split(',')]
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # 'Q' or 'M2'
        time_index = time_index[time_index.month.isin(mon_list)]
    
    elif time_freq == 'M1':
        s = valid_times.split(',')[0]
        e = valid_times.split(',')[1]
        s = pd.to_datetime(s,format='%Y%m')
        e = pd.to_datetime(e,format='%Y%m') + pd.DateOffset(months=1)
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1] # M1
    
    elif time_freq == 'D1':
        s = valid_times.split(',')[0]
        e = valid_times.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D') # D1
    
    elif time_freq == 'D2': # ['%Y,%Y','%m%d,%m%d']
        s = valid_times[0].split(',')[0]
        e = valid_times[1].split(',')[1]
        s_mon = valid_times[1].split(',')[0][:2]
        e_mon = valid_times[1].split(',')[1][:2]
        s_day = valid_times[1].split(',')[0][2:]
        e_day = valid_times[1].split(',')[1][2:]
        dates = pd.date_range(start=s, end=e, freq='D')
        time_index = dates[((dates.month==s_mon) & (dates.day>=s_day)) | ((dates.month>s_mon) & (dates.month<e_mon)) | ((dates.month==e_mon) & (dates.day<=e_day))]

    time_index = time_index[~((time_index.month==2) & (time_index.day==29))] # 由于数据原因，删除2月29号
    hydro_lon = data_df['Lon'][0]
    hydro_lat = data_df['Lat'][0]
    for _, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti]['tmp']
        for _, sub_dict2 in sub_dict1.items():
            for key, ds_data in sub_dict2.items():
                try:
                    selected_data = ds_data.sel(time=time_index)
                except:
                    selected_data = ds_data
                # selected_data = selected_data.interp(lat=hydro_lat, lon=hydro_lon, method='nearest')
                sub_dict2[key] = selected_data
        
    # 结果计算
    mon_dict = dict()
    # 1.水文站数据的原始统计结果
    result_q = stats_q(data_df, refer_df)
    result_q = result_q.to_dict(orient='records')

    # 2.模拟(观测) 使用验证期的气象数据计算径流
    data_df_meteo['EVP_Taka'] = data_df_meteo['EVP_Taka'].apply(lambda x: 0 if x<0 else x)
    data_df_meteo['PRE_Time_2020'] = data_df_meteo['PRE_Time_2020'].fillna(0)
    data_df_meteo['EVP_Taka'] = data_df_meteo['EVP_Taka'].fillna(0)
    data_df_meteo.dropna(how='any', axis=0, inplace=True)

    tem_daily = data_df_meteo.pivot_table(index=data_df_meteo.index, columns=['Station_Id_C'], values='TEM_Avg')  # 统计时段df
    tem_daily = tem_daily.mean(axis=1) # 区域平均，代表流域的平均情况
    tem_monthly = tem_daily.resample('1M').mean()
    
    pre_daily = data_df_meteo.pivot_table(index=data_df_meteo.index, columns=['Station_Id_C'], values='PRE_Time_2020')  # 统计时段df
    pre_daily = pre_daily.mean(axis=1)
    
    evp_daily = data_df_meteo.pivot_table(index=data_df_meteo.index, columns=['Station_Id_C'], values='EVP_Taka')  # 统计时段df
    evp_daily = evp_daily.mean(axis=1)
    evp_monthly = evp_daily.resample('1M').mean()
    
    # hbv-input
    date_time = tem_daily.index
    month = tem_daily.index.month.values
    temp = tem_daily.values  # 气温 单位：度
    precip = pre_daily.values  # 单位：mm
    
    q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
    q_sim = pd.DataFrame(q_sim, index=tem_daily.index, columns=['Q'])
    q_sim_yearly = q_sim.resample('1A').mean().round(2) # m^3/s
    q_sim_yearly1 = stats_result_1(q_sim_yearly, refer_df)
    
    mon_dict['验证期气象数据径流'] = q_sim.resample('1M').mean().round(1)

    # 3.模拟（模式） 使用验证期的cmip6数据计算径流 
    # 同一情境 不同模式集合平均
    vaild_cmip_res = dict()
    for exp, sub_dict1 in vaild_cmip.items():  # vaild_cmip[exp][insti]['tmp']
        tmp_list = []
        pre_list = []
        
        mon_dict[exp] = dict()
        
        for insti, sub_dict2 in sub_dict1.items():
            tmp = sub_dict2['tmp']
            pre = sub_dict2['pre']
            tmp_list.append(tmp)
            pre_list.append(pre)
        
        tem_daily = xr.concat(tmp_list, 'new_dim')
        tem_daily = tem_daily.mean(dim='new_dim')
        pre_daily = xr.concat(pre_list, 'new_dim')
        pre_daily = pre_daily.mean(dim='new_dim')
        
        # 数据处理
        tem_monthly = tem_daily.resample(time='1M').mean()
        tem_monthly = tem_monthly.tas.to_series()
        pre_monthly = pre_daily.resample(time='1M').sum()
        pre_monthly = pre_monthly.pr.to_series()
        evp_monthly = 3100*tem_monthly/(3100+1.8*(pre_monthly**2)*np.exp((-34.4*tem_monthly)/(235+tem_monthly)))
        evp_monthly = evp_monthly.where(evp_monthly>0,0)
        
        # hbv-input
        date_time = pd.to_datetime(tem_daily.time)
        month = np.array(pd.to_datetime(tem_daily.time).month)
        temp = tem_daily.tas.data  # 气温 单位：度
        precip = pre_daily.pr.data  # 单位：mm
        
        q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
        q_sim_1 = pd.DataFrame(q_sim, index=pd.to_datetime(tem_daily.time), columns=['Q'])
        q_sim_yearly = q_sim_1.resample('1A').mean().round(2)
        vaild_cmip_res[exp] = q_sim_yearly
        
        mon_dict[exp]['验证期模式数据径流'] = q_sim_1.resample('1M').mean().round(1)
        
    vaild_cmip_res = stats_result_2(vaild_cmip_res, refer_df)
    
    # 4.预估-集合模式
    evaluate_cmip_res = dict()
    for exp, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti]['tmp']
        tmp_list = []
        pre_list = []
        for insti, sub_dict2 in sub_dict1.items():
            tmp = sub_dict2['tmp']
            pre = sub_dict2['pre']
            tmp_list.append(tmp)
            pre_list.append(pre)
        
        tem_daily = xr.concat(tmp_list, 'new_dim')
        tem_daily = tem_daily.mean(dim='new_dim')
        pre_daily = xr.concat(pre_list, 'new_dim')
        pre_daily = pre_daily.mean(dim='new_dim')
        
        # 数据处理
        tem_monthly = tem_daily.resample(time='1M').mean()
        tem_monthly = tem_monthly.tas.to_series()
        pre_monthly = pre_daily.resample(time='1M').sum()
        pre_monthly = pre_monthly.pr.to_series()
        evp_monthly = 3100*tem_monthly/(3100+1.8*(pre_monthly**2)*np.exp((-34.4*tem_monthly)/(235+tem_monthly)))
        evp_monthly = evp_monthly.where(evp_monthly>0,0)
        
        # hbv-input
        date_time = pd.to_datetime(tem_daily.time)
        month = np.array(pd.to_datetime(tem_daily.time).month)
        temp = tem_daily.tas.data  # 气温 单位：度
        precip = pre_daily.pr.data  # 单位：mm
        q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
        q_sim_2 = pd.DataFrame(q_sim, index=pd.to_datetime(tem_daily.time), columns=['Q'])
        q_sim_yearly = q_sim_2.resample('1A').mean().round(2)
        evaluate_cmip_res[exp] = q_sim_yearly

        mon_dict[exp]['预估集合径流'] = q_sim_2.resample('1M').mean().round(1)

    evaluate_cmip_res = stats_result_2(evaluate_cmip_res, refer_df)

    # 5.预估-单情景-单模式
    single_cmip_res = dict()
    for exp, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti]['tmp']
        single_cmip_res[exp] = dict()
        for insti, sub_dict2 in sub_dict1.items():
            tem_daily = sub_dict2['tmp']
            pre_daily = sub_dict2['pre']
        
            # 数据处理
            tem_monthly = tem_daily.resample(time='1M').mean()
            tem_monthly = tem_monthly.tas.to_series()
            pre_monthly = pre_daily.resample(time='1M').sum()
            pre_monthly = pre_monthly.pr.to_series()
            evp_monthly = 3100*tem_monthly/(3100+1.8*(pre_monthly**2)*np.exp((-34.4*tem_monthly)/(235+tem_monthly)))
            evp_monthly = evp_monthly.where(evp_monthly>0,0)
            
            # hbv-input
            date_time = pd.to_datetime(tem_daily.time)
            month = np.array(pd.to_datetime(tem_daily.time).month)
            temp = tem_daily.tas.data  # 气温 单位：度
            precip = pre_daily.pr.data  # 单位：mm
            q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
            q_sim_3 = pd.DataFrame(q_sim, index=pd.to_datetime(tem_daily.time), columns=['Q'])
            q_sim_yearly = q_sim_3.resample('1A').mean().round(2)
            single_cmip_res[exp][insti] = q_sim_yearly
            
            mon_dict[exp][insti] = q_sim_3.resample('1M').mean().round(1)
            
    single_cmip_res = stats_result_3(single_cmip_res, refer_df)
    
    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['表格历史'] = dict()
    result_dict['表格预估'] = dict()
    result_dict['表格历史']['观测'] = result_q
    result_dict['表格历史']['模拟观测'] = q_sim_yearly1.to_dict(orient='records')
    result_dict['表格历史']['模拟模式'] = vaild_cmip_res.to_dict(orient='records')
    result_dict['表格预估']['集合'] = evaluate_cmip_res.to_dict(orient='records')
    result_dict['表格预估']['单模式'] = single_cmip_res.to_dict(orient='records')
    
    # 4.时序图-各个情景的集合
    std_percent = dict()
    for exp, sub_dict in single_cmip_res.items():
        std_percent[exp] = dict()
        array_list= []
        for insti, res_df in sub_dict.items():
            res_df = pd.DataFrame(res_df)
            res_df.set_index('index',inplace=True)
            array_list.append(res_df.iloc[:-7, 2].values[None])
            array = np.concatenate(array_list,axis=0)
        

        std = np.std(array, ddof=1, axis=0).round(2)
        per25 = np.percentile(array, 25, axis=0).round(2)
        per75 = np.percentile(array, 75, axis=0).round(2)
        
        std = pd.DataFrame(std, index=res_df.index[:-7], columns=[res_df.columns[2]])
        per25 = pd.DataFrame(per25, index=res_df.index[:-7], columns=[res_df.columns[2]])
        per75 = pd.DataFrame(per75, index=res_df.index[:-7], columns=[res_df.columns[2]])
        
        std.reset_index(drop=False,inplace=True)
        per25.reset_index(drop=False,inplace=True)
        per75.reset_index(drop=False,inplace=True)
        
        std_percent[exp]['1倍标准差'] = std.to_dict(orient='records')
        std_percent[exp]['百分位数25'] = per25.to_dict(orient='records')
        std_percent[exp]['百分位数75'] = per75.to_dict(orient='records')
    
    result_dict['时序图'] = std_percent

    return result_dict, mon_dict

    
    # data_df 验证期水文数据
    # refer_df 参考时段水文数据
    # data_df_meteo 验证期气象数据（站点平均）
    # vaild_cmip 验证期cmip数据，插值到水文站
    # evaluate_cmip 预估期cmip数据，插值到水文站
    # result_q 水文站数据观测结果
    # q_sim_yearly 验证期气象站数据的HBV结果
    # vaild_cmip_res 验证期cmip数据的HBV结果
    # evaluate_cmip_res 预估期cmip数据的HBV结果（集合）
    # single_cmip_res 预估器cmip数据的HBV结果（单一模式）
    # return data_df, refer_df, data_df_meteo, vaild_cmip, evaluate_cmip, result_q, q_sim_yearly, vaild_cmip_res, evaluate_cmip_res, single_cmip_res


if __name__ == '__main__':
    data_json = dict()
    data_json['time_freq'] = 'Y'
    data_json['evaluate_times'] = "2023,2050" # 预估时段时间条
    data_json['refer_years'] = '2023,2024'# 参考时段时间条
    data_json['valid_times'] = '202303,202403' # 验证期 '%Y%m,%Y%m'
    data_json['hydro_ids'] = '40100350' # 唐乃亥
    data_json['sta_ids'] = "52943,52957,52955,56033,56067,56045,56046,56043,56065,52968,56074,56079,56173"
    data_json['cmip_type'] = 'original' # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ['Set']# 模式，列表：['CanESM5','CESM2']等
    data_json['d'] = 6.1
    data_json['fc'] = 195
    data_json['beta'] = 2.6143
    data_json['c'] = 0.07
    data_json['k0'] = 0.163
    data_json['l'] = 4.87
    data_json['k1'] = 0.027
    data_json['k2'] = 0.049
    data_json['kp'] = 0.05
    data_json['pwp'] = 106
    data_json['Tsnow_thresh'] = 0
    data_json['ca'] = 50000
    # ddata_df, refer_df, data_df_meteo, vaild_cmip, evaluate_cmip, result_q, q_sim_yearly, vaild_cmip_res, evaluate_cmip_res, single_cmip_res = hbv_single_calc(data_json)
    result_dict, mon_dict = hbv_single_calc(data_json)