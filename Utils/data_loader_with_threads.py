import os
import platform
import glob
import time
import logging
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.config import cfg
from libs.nmc_met_io.retrieve_cmadaas import cmadaas_obs_by_time_range_and_id
from libs.nmc_met_io.retrieve_cmadaas import cmadaas_obs_by_period_and_id
import psycopg2
from psycopg2 import sql


def get_cmadaas_yearly_data(years, elements, sta_ids):
    '''
    多线程年值数据下载
    '''

    def download_year(vals, ranges=None):
        '''
        数据下载代码
        '''
        time_range, elements, sta_ids = vals
        data_code = 'SURF_CHN_MUL_YER'
        default_elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,Year'
        all_elements = default_elements + ',' + elements
        df = cmadaas_obs_by_time_range_and_id(time_range=time_range, data_code=data_code, elements=all_elements, ranges=ranges, sta_ids=sta_ids)
        return df

    # 生成输入参数
    start_year, end_year = map(int, years.split(','))
    range_year = np.arange(start_year, end_year + 1, 1)
    time_range_template = '[{},{}]'
    all_params = [(time_range_template.format(str(year) + '0101000000', str(year) + '1231230000'), elements, sta_ids) for year in range_year]

    # 创建线程池
    with ThreadPoolExecutor(max_workers=cfg.INFO.NUM_THREADS) as pool:
        futures = [pool.submit(download_year, param) for param in all_params]

    # 获取结果并合并数据
    # dfs = [f.result() for f in as_completed(futures)]
    dfs = []
    for f in as_completed(futures):
        if f.result() is not None:
            tmp = f.result().drop_duplicates()
            dfs.append(tmp)

    # concentrate dataframes
    if len(dfs) == 0:
        return None
    else:
        return pd.concat(dfs, axis=0, ignore_index=True).sort_values(by='Datetime')


def get_cmadaas_monthly_data(years, mon, elements, sta_ids):
    '''
    多线程月值数据下载
    '''

    def download_month(vals, ranges=None):
        '''
        数据下载代码
        '''
        time_range, elements, sta_ids = vals
        data_code = 'SURF_CHN_MUL_MON'
        default_elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,Year,Mon'
        all_elements = default_elements + ',' + elements
        ranges = None
        df = cmadaas_obs_by_time_range_and_id(time_range=time_range, data_code=data_code, elements=all_elements, ranges=ranges, sta_ids=sta_ids)
        return df

    # 生成输入参数
    start_year, end_year = map(int, years.split(','))
    start_month = mon.split(',')[0]
    end_month = mon.split(',')[1]
    range_year = np.arange(start_year, end_year + 1, 1)
    time_range_template = '[{},{}]'
    all_params = [(time_range_template.format(str(year) + start_month + '01000000', str(year) + end_month + '31230000'), elements, sta_ids) for year in range_year]

    # 创建线程池
    with ThreadPoolExecutor(max_workers=cfg.INFO.NUM_THREADS) as pool:
        futures = [pool.submit(download_month, param) for param in all_params]

    # 获取结果并合并数据
    # dfs = [f.result() for f in as_completed(futures)]
    dfs = []
    for f in as_completed(futures):
        if f.result() is not None:
            tmp = f.result().drop_duplicates()
            dfs.append(tmp)

    # concentrate dataframes
    if len(dfs) == 0:
        return None
    else:
        return pd.concat(dfs, axis=0, ignore_index=True).sort_values(by='Datetime')


def get_cmadaas_daily_data(years, date, elements, sta_ids):
    '''
    多线程日值数据下载
    '''

    def download_day(vals, ranges=None):
        '''
        数据下载代码
        '''
        time_range, elements, sta_ids = vals
        data_code = 'SURF_CHN_MUL_DAY'
        default_elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,Year,Mon,Day'
        all_elements = default_elements + ',' + elements
        df = cmadaas_obs_by_time_range_and_id(time_range=time_range, data_code=data_code, elements=all_elements, ranges=ranges, sta_ids=sta_ids)
        return df

    # 生成输入参数
    start_year, end_year = map(int, years.split(','))
    range_year = np.arange(start_year, end_year + 1, 1)
    start_date = date.split(',')[0]
    end_date = date.split(',')[1]
    time_range_template = '[{},{}]'
    all_params = [(time_range_template.format(str(year) + start_date + '000000', str(year) + end_date + '230000'), elements, sta_ids) for year in range_year]

    # 创建线程池
    with ThreadPoolExecutor(max_workers=cfg.INFO.NUM_THREADS) as pool:
        futures = [pool.submit(download_day, param) for param in all_params]

    # 获取结果并合并数据
    # dfs = [f.result() for f in as_completed(futures)]
    dfs = []
    for f in as_completed(futures):
        if f.result() is not None:
            tmp = f.result().drop_duplicates()
            dfs.append(tmp)

    # concentrate dataframes
    if len(dfs) == 0:
        return None
    else:
        return pd.concat(dfs, axis=0, ignore_index=True).sort_values(by='Datetime')


def get_cmadaas_daily_period_data(years, date, elements, sta_ids):
    '''
    多线程日值数据下载,下载历年时段
    '''

    def download_day(vals, ranges=None):
        '''
        数据下载代码
        '''
        minYear, maxYear, minMD, maxMD, elements, sta_ids = vals
        data_code = 'SURF_CHN_MUL_DAY'
        default_elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,Year,Mon,Day'
        all_elements = default_elements + ',' + elements
        df = cmadaas_obs_by_period_and_id(minYear, maxYear, minMD, maxMD, data_code=data_code, elements=all_elements, ranges=ranges, sta_ids=sta_ids)
        return df

    # 生成输入参数
    start_year, end_year = map(int, years.split(','))
    range_year = np.arange(start_year, end_year + 1, 1)
    start_date = date.split(',')[0]
    end_date = date.split(',')[1]
    all_params = [(year, year+1, start_date, end_date, elements, sta_ids) for year in range_year]

    # 创建线程池
    with ThreadPoolExecutor(max_workers=cfg.INFO.NUM_THREADS) as pool:
        futures = [pool.submit(download_day, param) for param in all_params]

    # 获取结果并合并数据
    # dfs = [f.result() for f in as_completed(futures)]
    dfs = []
    for f in as_completed(futures):
        if f.result() is not None:
            tmp = f.result().drop_duplicates()
            dfs.append(tmp)

    # concentrate dataframes
    if len(dfs) == 0:
        return None
    else:
        return pd.concat(dfs, axis=0, ignore_index=True).sort_values(by='Datetime')
    


def get_database_data(sta_ids, element_str, table_name, time_freq, stats_times):
    '''
    从数据库获取数据
    '''
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
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
    df = pd.DataFrame(data)
    df.columns = elements.split(',')
    
    # 关闭数据库
    cur.close()
    conn.close()

    return df