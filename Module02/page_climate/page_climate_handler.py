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

# 气候要素预估接口


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

def climate_forcast(data_json):
    '''
    获取天擎数据，参数说明

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
    
    :param refer_years: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'

    :param sta_ids: 传入的气象站点

    :param element：对应原型，传入的要素名称
        平均气温	TEM_Avg 
        最高气温	TEM_Max
        最低气温	TEM_Min
        降水量	PRE_Time_2020
        降水日数	PRE_Days
        平均风速	WIN_S_2mi_Avg
        平均相对湿度	RHU_Avg
    '''
    # 1.参数读取
    time_freq = data_json['time_freq'] # 控制预估时段
    stats_times = data_json['stats_times'] # 预估时段时间条
    refer_years = data_json['refer_years'] # 参考时段时间条
    element = data_json['element']
    sta_ids = data_json['sta_ids'] # 气象站 '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    cmip_type = data_json['cmip_type'] # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    cmip_res = data_json.get('cmip_res') # 分辨率 1/5/10/25/50/100 km
    cmip_model = data_json['cmip_model'] # 模式，列表：['CanESM5','CESM2']等

    inpath = '/zipdata'
    # inpath = r'C:\Users\MJY\Desktop\qhbh\zipdata\cmip6' # cmip6路径

    # 2.参数处理
    degree = None
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)

    ######################################################
    # 站点数据获取
    if time_freq == 'Y':
        table_name = 'qh_qhbh_cmadaas_year'
    elif time_freq in ['Q', 'M1', 'M2']:
        table_name = 'qh_qhbh_cmadaas_month'
    elif time_freq in ['D1', 'D2']:
        table_name = 'qh_qhbh_cmadaas_day'
    element_str = element
    
    # 从数据库截数据
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
    sta_ids = tuple(sta_ids.split(','))
    elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,' + element_str

    if time_freq == 'Y':  # '%Y,%Y'
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND station_id_c IN %s
                        """)

        # 根据sql获取统计年份data
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        cur.execute(query, (start_year, end_year, sta_ids))
        data = cur.fetchall()

    elif time_freq == 'Q':  # ['%Y,%Y','3,4,5']
        mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
        mon_ = tuple(mon_list)
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]

        if 12 in mon_list:
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                                WHERE (SUBSTRING(datetime, 1, 4) BETWEEN %s AND %s) 
                                AND SUBSTRING(datetime, 6, 2) IN ('12', '01', '02')
                                OR (SUBSTRING(datetime, 1, 4) = %s AND SUBSTRING(datetime, 6, 2) = '12')
                                OR (SUBSTRING(datetime, 1, 4) = %s AND SUBSTRING(datetime, 6, 2) IN ('01', '02'))
                                AND station_id_c IN %s
                            """)
            cur.execute(query, (start_year, end_year,str(int(start_year)-1),str(int(end_year)+1), sta_ids))

        else:    
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                            WHERE
                                (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                                AND station_id_c IN %s
                            """)  
            cur.execute(query, (start_year, end_year, mon_, sta_ids))

        data = cur.fetchall()

    elif time_freq == 'M1':  # '%Y%m,%Y%m'
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s))
                            AND station_id_c IN %s
                        """)

        start_year = stats_times.split(',')[0][:4]
        end_year = stats_times.split(',')[1][:4]
        start_month = stats_times.split(',')[0][4:]
        end_month = stats_times.split(',')[1][4:]
        cur.execute(query, (start_year, start_month, start_year, end_year, end_year, end_month, sta_ids))
        data = cur.fetchall()

    elif time_freq == 'M2':  # ['%Y,%Y','11,12,1,2']
        mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
        mon_ = tuple(mon_list)
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                            AND station_id_c IN %s
                        """)

        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        cur.execute(query, (start_year, end_year, mon_, sta_ids))
        data = cur.fetchall()

    elif time_freq == 'D1':  # '%Y%m%d,%Y%m%d'
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s))
                            AND station_id_c IN %s
                        """)

        start_year = stats_times.split(',')[0][:4]
        end_year = stats_times.split(',')[1][:4]
        start_month = stats_times.split(',')[0][4:6]
        end_month = stats_times.split(',')[1][4:6]
        start_date = stats_times.split(',')[0][6:]
        end_date = stats_times.split(',')[1][6:]
        cur.execute(query, (start_year, start_month, start_date, start_year, end_year, end_year, end_month, end_date, sta_ids))
        data = cur.fetchall()

    elif time_freq == 'D2':  # ['%Y,%Y','%m%d,%m%d']
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND (
                                (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                                OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) < %s)
                                OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s)
                            ))
                            AND station_id_c IN %s
                        """)

        years = stats_times[0]
        dates = stats_times[1]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        start_mon = dates.split(',')[0][:2]
        end_mon = dates.split(',')[1][:2]
        start_date = dates.split(',')[0][2:]
        end_date = dates.split(',')[1][2:]
        cur.execute(query, (start_year, end_year, start_mon, start_date, start_mon, end_mon, end_mon, end_date, sta_ids))
        data = cur.fetchall()

    # 统计年份数据处理为df
    data_df = pd.DataFrame(data)
    data_df.columns = elements.split(',')
    
    # 下载参考时段的数据
    query = sql.SQL(f"""
                    SELECT {elements}
                    FROM public.{table_name}
                    WHERE
                        CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                        AND station_id_c IN %s
                    """)

    start_year = refer_years.split(',')[0]
    end_year = refer_years.split(',')[1]
    cur.execute(query, (start_year, end_year, sta_ids))
    data = cur.fetchall()
    refer_df = pd.DataFrame(data)
    refer_df.columns = elements.split(',')

    # 关闭数据库
    cur.close()
    conn.close()

    ######################################################
    # 模式数据获取
    # 先确定年份
    if time_freq == 'Y':  # '%Y,%Y'
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]

    elif time_freq in ['Q', 'M2', 'D2']:  # ['%Y,%Y','3,4,5']
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]

    elif time_freq in ['M1', 'D1']:  # '%Y%m,%Y%m'
        start_year = stats_times.split(',')[0][:4]
        end_year = stats_times.split(',')[1][:4]

    # 确定模式原始要素
    var_dict = dict()
    var_dict['TEM_Avg'] = 'tas'
    var_dict['TEM_Max'] = 'tasmax'
    var_dict['TEM_Min'] = 'tasmin'
    var_dict['PRE_Time_2020'] = 'pr'
    var_dict['PRE_Days'] = 'pr'
    var_dict['RHU_Avg'] = 'hurs'
    # var_dict['WIN_S_2mi_Avg'] = 'uas,vas'
    var = var_dict['elements']

    # 读取数据
    evaluate_cmip = dict()
    for exp in ['ssp126','ssp245']:
        evaluate_cmip[exp] = dict()
        for insti in cmip_model:
            evaluate_cmip[exp][insti] = dict()
            tmp_lst = []
            for year in range(start_year,end_year+1):
                tem_file_path = choose_mod_path(inpath=inpath, 
                                                data_source=cmip_type,
                                                insti=insti, 
                                                var=var, 
                                                time_scale='daily', 
                                                yr=year, 
                                                expri_i=exp, 
                                                res=cmip_res)

                ds_tmp = xr.open_dataset(tem_file_path)
                tmp_lst.append(ds_tmp)
            
            tmp_all = xr.concat(tmp_lst, dim='time')
            tmp_all['time'] = tmp_all.indexes['time'].to_datetimeindex().normalize()
            evaluate_cmip[exp][insti][var] = tmp_all

    ######################################################
    # 数据处理
    ##### 站点数据
    data_df = data_processing(data_df, element_str, degree)
    refer_df = data_processing(refer_df, element_str, degree)


    ##### 预估期的cmip6插值到水文站
    # 首先筛选时间
    if time_freq == 'Y':
        s = stats_times.split(',')[0]
        e = stats_times.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1] # 'Y'

    elif time_freq in ['Q', 'M2']:
        s = stats_times[0].split(',')[0]
        e = stats_times[1].split(',')[1]
        mon_list = [int(val) for val in stats_times[1].split(',')]
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # 'Q' or 'M2'
        time_index = time_index[time_index.month.isin(mon_list)]
    
    elif time_freq == 'M1':
        s = stats_times.split(',')[0]
        e = stats_times.split(',')[1]
        s = pd.to_datetime(s,format='%Y%m')
        e = pd.to_datetime(e,format='%Y%m') + pd.DateOffset(months=1)
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1] # M1
    
    elif time_freq == 'D1':
        s = stats_times.split(',')[0]
        e = stats_times.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D') # D1
    
    elif time_freq == 'D2': # ['%Y,%Y','%m%d,%m%d']
        s = stats_times[0].split(',')[0]
        e = stats_times[1].split(',')[1]
        s_mon = stats_times[1].split(',')[0][:2]
        e_mon = stats_times[1].split(',')[1][:2]
        s_day = stats_times[1].split(',')[0][2:]
        e_day = stats_times[1].split(',')[1][2:]
        dates = pd.date_range(start=s, end=e, freq='D')
        time_index = dates[((dates.month==s_mon) & (dates.day>=s_day)) | ((dates.month>s_mon) & (dates.month<e_mon)) | ((dates.month==e_mon) & (dates.day<=e_day))]

    time_index = time_index[~((time_index.month==2) & (time_index.day==29))] # 由于数据原因，删除2月29号
    hydro_lon = data_df['Lon'][0]
    hydro_lat = data_df['Lat'][0]
    for _, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti]['tmp']
        for _, sub_dict2 in sub_dict1.items():
            for key, ds_data in sub_dict2.items():
                selected_data = ds_data.sel(time=time_index)
                selected_data = selected_data.interp(lat=hydro_lat, lon=hydro_lon, method='nearest')
                sub_dict2[key] = selected_data


 


if __name__ == '__main__':
    data_json = dict()
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '1950,1980' # 预估时段时间条
    data_json['refer_years'] = '2023,2024'# 参考时段时间条
    data_json['sta_ids'] = '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    data_json['cmip_type'] = 'original' # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ['BCC-CSM2-MR', 'CanESM5']# 模式，列表：['CanESM5','CESM2']等

    # ddata_df, refer_df, data_df_meteo, vaild_cmip, evaluate_cmip, result_q, q_sim_yearly, vaild_cmip_res, evaluate_cmip_res, single_cmip_res = hbv_single_calc(data_json)
    result_dict = hbv_single_calc(data_json)
