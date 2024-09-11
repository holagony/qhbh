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

    if insti == 'CNRM-CM6-1':
        data_grid = '_r1i1p1f2_gr_'
    elif (insti == 'BCC-CSM2-MR') & (yr < 2015):
        data_grid = '_r3i1p1f1_gn_'
    else:
        data_grid = '_r1i1p1f1_gn_'

    if time_scale == 'daily':
        path1 = 'daily'
        filen = var + '_day_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    elif time_scale == 'monthly':
        path1 = 'monthly'
        filen = var + '_month_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    elif time_scale == 'yearly':
        path1 = 'yearly'
        filen = var + '_year_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    else:
        path1 = time_scale
        filen = var + '_' + time_scale + '_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'

    if data_source == 'original':
        path = os.path.join(inpath, data_source, path1, insti, expri, var, filen)
    else:
        path = os.path.join(inpath, data_source, res, path1, insti, expri, var, filen)

    return path

# inpath = '/share/data5/QH_qihou_proj-200G/data/cmip_data/'
# data_source = 'original'
# insti = 'BCC-CSM2-MR'
# var = 'pr-new'
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
    valid_times = data_json['valid_times'] # 率定期 '%Y%m,%Y%m'
    hydro_ids = data_json['hydro_ids'] # 水文站 40100350 唐乃亥
    sta_ids = data_json['sta_ids'] # 水文站对应的气象站 唐乃亥对应 '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    cmip_type = data_json['cmip_type'] # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    cmip_res = data_json.get('cmip_res') # 分辨率 1/5/10/25/50/100 km
    cmip_model = data_json['cmip_model'] # 模式，列表：['CanESM5','CESM2']等
    degree = data_json.get('degree')

    # 2.参数处理
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)

    ######################################################
    # 数据下载
    ##### 下载验证期时段 & 参考时段的水文数据 （对应表格-观测）
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
    cur.execute(query, (start_year, start_month, start_year, end_year, end_year, end_month, sta_ids))
    data = cur.fetchall()
    data_df = pd.DataFrame(data)
    data_df.columns = elements.split(',')

    # 参考时段的水文站数据 计算距平
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

    ##### 下载验证期时段相应的气象数据，并处理，用于HBV计算 （对应表格-模拟（观测））
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

    ##### 下载验证期时段的cmip6数据，并插值到站点，用于HBV计算，（对应表格-模拟（模式）） 
    start_year = int(valid_times.split(',')[0][:4])
    end_year = int(valid_times.split(',')[1][:4])
    inpath = '/zipdata/QH_qihou_proj-200G/data/cmip_data' # 挂载路径

    file_path = dict() 
    for exp in ['ssp126','ssp245','ssp585']:
        file_path[exp] = dict()
        for insti in cmip_model:
            file_path[exp][insti] = dict()
            for year in range(start_year,end_year+1):
                file_path[exp][insti][year] = dict()
                tem_file_path = choose_mod_path(inpath=inpath, 
                                                data_source=cmip_type, 
                                                insti=insti, 
                                                var='tas', 
                                                time_scale='daily', 
                                                yr=year, 
                                                expri_i=exp, 
                                                res=cmip_res)

                pre_file_path = choose_mod_path(inpath=inpath, 
                                                data_source=cmip_type, 
                                                insti=insti, 
                                                var='pr', 
                                                time_scale='daily', 
                                                yr=year, 
                                                expri_i=exp, 
                                                res=cmip_res)
                
                file_path[exp][insti][year]['tem'] = tem_file_path
                file_path[exp][insti][year]['pre'] = pre_file_path

    ##### 下载预估时段的cmip6数据，并插值到站点，用于HBV计算，使用预估时间（在这个里面生成预估气象数据，对应预估）
    '''
    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'
        季 - 'Q'
        月(连续) - 'M1'
        月(区间) - 'M2' 
        日(连续) - 'D1'
        日(区间) - 'D2'

    :param stats_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'
        (2)当time_freq选择季Q。下载连续的月数据，处理成季数据（多下两个月数据），提取历年相应的季节数据，传：['%Y,%Y','3,4,5']，其中'3,4,5'为季度对应的月份 
        (3)当time_freq选择月(连续)M1。下载连续的月数据，传参：'%Y%m,%Y%m'
        (4)当time_freq选择月(区间)M2。下载连续的月数据，随后提取历年相应的月数据，传参：['%Y,%Y','11,12,1,2'] 前者年份，'11,12,1,2'为连续月份区间
        (5)当time_freq选择日(连续)D1。下载连续的日数据，传参：'%Y%m%d,%Y%m%d'
        (6)当time_freq选择日(区间)D2。直接调天擎接口，下载历年区间时间段内的日数据，传：['%Y,%Y','%m%d,%m%d'] 前者年份，后者区间
    '''
    if time_freq == 'Y':
        start_year = int(evaluate_times.split(',')[0])
        end_year = int(evaluate_times.split(',')[1])
    elif time_freq in ['Q', 'M2', 'D2']:
        start_year = int(evaluate_times[0].split(',')[0])
        end_year = int(evaluate_times[1].split(',')[1])
    elif time_freq in ['M1', 'D1']:
        start_year = int(evaluate_times.split(',')[0][:4])
        end_year = int(evaluate_times.split(',')[1][:4])

    inpath = '/zipdata/QH_qihou_proj-200G/data/cmip_data' # 挂载路径

    file_path_1 = dict() 
    for exp in ['ssp126','ssp245','ssp585']:
        file_path_1[exp] = dict()
        for insti in cmip_model:
            file_path_1[exp][insti] = dict()
            for year in range(start_year,end_year+1):
                file_path_1[exp][insti][year] = dict()
                tem_file_path = choose_mod_path(inpath=inpath, 
                                                data_source=cmip_type, 
                                                insti=insti, 
                                                var='tas', 
                                                time_scale='daily', 
                                                yr=year, 
                                                expri_i=exp, 
                                                res=cmip_res)

                pre_file_path = choose_mod_path(inpath=inpath, 
                                                data_source=cmip_type, 
                                                insti=insti, 
                                                var='pr', 
                                                time_scale='daily', 
                                                yr=year, 
                                                expri_i=exp, 
                                                res=cmip_res)

                file_path_1[exp][insti][year]['tem'] = tem_file_path
                file_path_1[exp][insti][year]['pre'] = pre_file_path

    ######################################################
    # 数据处理
    ##### 水文数据处理
    data_df['Datetime'] = pd.to_datetime(data_df['Datetime'],format='%Y%m%d')
    data_df.set_index('Datetime', inplace=True)
    data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    data_df['Lon'] = data_df['Lon'].astype(float)
    data_df['Lat'] = data_df['Lat'].astype(float)
    data_df['Q'] = data_df['Q'].astype(float) # 日尺度

    refer_df['Datetime'] = pd.to_datetime(refer_df['Datetime'],format='%Y%m%d')
    refer_df.set_index('Datetime', inplace=True)
    refer_df['Station_Id_C'] = refer_df['Station_Id_C'].astype(str)
    refer_df['Lon'] = refer_df['Lon'].astype(float)
    refer_df['Lat'] = refer_df['Lat'].astype(float)
    refer_df['Q'] = refer_df['Q'].astype(float) # 日尺度

    ##### 验证期的气象数据
    data_df_meteo['Datetime']

































    # 数据处理













    return result_dict


if __name__ == '__main__':
    t1 = time.time()
    data_json = dict()
    data_json['element'] = 'Rad'
    data_json['refer_years'] = '2023,2025'
    data_json['nearly_years'] = '2023,2025'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2023,2025'  # '198105,202009' # '1981,2023'
    data_json['sta_ids'] = '52866,56029,52863,52754,52818,52874,56043,52713,56065'
    data_json['interp_method'] = 'idw2'
    data_json['ci'] = 95
    data_json['shp_path'] = r'C:\Users\MJY\Desktop\qhbh\文档\03-边界矢量\03-边界矢量\03-边界矢量\01-青海省\青海省县级数据.shp'
    data_json['degree'] = None
    
    result = climate_features_stats(data_json)
    t2 = time.time()
    print(t2 - t1)
