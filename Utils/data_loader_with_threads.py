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