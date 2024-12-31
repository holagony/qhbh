# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:38:06 2024

@author: EDY

模式数据路径拼接

根目录/时间尺度/模式名称/情景模式/元素名称
"""
import os
import glob
import pandas as pd
import xarray as xr
from Utils.config import cfg


def read_xlsx_data(file, station_id):
    data_1 = pd.read_csv(file)
    if 'Unnamed: 0' in data_1.columns:
        data_1.drop(['Unnamed: 0'], axis=1, inplace=True)

    data_1.iloc[0, 1::] = data_1.iloc[0, 1::].astype(int).astype(str)
    ori_columns = data_1.columns
    data_1.columns = data_1.loc[0]

    data_1 = data_1.copy()
    data_1.loc[0] = ori_columns
    common_strings = set(data_1.columns).intersection(set(station_id))
    formatted_common_strings = [string.lower().replace(" ", "") for string in common_strings]

    data_11 = data_1[['station id'] + formatted_common_strings]
    data_11.rename(columns={data_11.columns[0]: 'Datetime'}, inplace=True)
    data_11 = data_11.iloc[3::, :]
    data_11['Datetime'] = pd.to_datetime(data_11['Datetime'])
    data_11.set_index('Datetime', inplace=True)
    # data_11.drop(['Datetime'], axis=1, inplace=True)

    return data_11


def read_model_data(data_dir, time_scale, insti, scene, var, stats_times, time_freq, station_id):
    '''
    批量读取模式数据
    '''
    # 通过时间尺度，取出时间年月日：
    if time_freq == 'Y':
        # Y
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]

        if int(start_year) < 2015:
            file_1 = os.path.join(data_dir, time_scale, insti, 'historical', var + '.csv')
            data_1 = read_xlsx_data(file_1, station_id)
            data_m1 = data_1.loc[start_year:'2014']
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m2 = data_2.loc['2015':end_year]
            data_m3 = pd.concat([data_m1, data_m2], axis=0)
        else:
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m3 = data_2.loc[start_year:end_year]

    elif time_freq in ['Q']:
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        month = stats_times[1]
        month = list(map(int, month.split(',')))

        if 12 in month:
            if int(start_year) < 2016:
                file_1 = os.path.join(data_dir, time_scale, insti, 'historical', var + '.csv')
                data_1 = read_xlsx_data(file_1, station_id)
                file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
                data_2 = read_xlsx_data(file_2, station_id)
                data_m1 = data_1[str(int(start_year) - 1) + '-12':'2014-12']
                data_m1 = data_m1[data_m1.index.month.isin(month)]
                data_m2 = data_2['2015-01':str(int(end_year) - 1) + '-02']
                data_m2 = data_m2[data_m2.index.month.isin(month)]
                data_m3 = pd.concat([data_m1, data_m2], axis=0)
            else:
                file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
                data_2 = read_xlsx_data(file_2, station_id)
                data_m2 = data_2[str(int(start_year) - 1) + '-12':str(int(end_year) - 1) + '-02']
                data_m2 = data_m2[data_m2.index.month.isin(month)]
                data_m3 = pd.concat([data_m1, data_m2], axis=0)
        else:
            if int(start_year) < 2015:
                file_1 = os.path.join(data_dir, time_scale, insti, 'historical', var + '.csv')
                data_1 = read_xlsx_data(file_1, station_id)
                file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
                data_2 = read_xlsx_data(file_2, station_id)
                data_m1 = data_1[data_1.index.month.isin(month)]
                data_m1 = data_m1.loc[start_year:'2014']
                data_m2 = data_2[data_2.index.month.isin(month)]
                data_m2 = data_m2.loc['2015':end_year]
                data_m3 = pd.concat([data_m1, data_m2], axis=0)
            else:
                file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
                data_2 = read_xlsx_data(file_2, station_id)
                data_m2 = data_2[data_2.index.month.isin(month)]
                data_m3 = data_m2.loc[start_year:end_year]

    elif time_freq in ['M2']:
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        month = stats_times[1]
        month = list(map(int, month.split(',')))

        if int(start_year) < 2015:
            file_1 = os.path.join(data_dir, time_scale, insti, 'historical', var + '.csv')
            data_1 = read_xlsx_data(file_1, station_id)
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m1 = data_1[data_1.index.month.isin(month)]
            data_m1 = data_m1.loc[start_year:'2014']
            data_m2 = data_2[data_2.index.month.isin(month)]
            data_m2 = data_m2.loc['2015':end_year]
            data_m3 = pd.concat([data_m1, data_m2], axis=0)
        else:
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m2 = data_2[data_2.index.month.isin(month)]
            data_m3 = data_m2.loc[start_year:end_year]

    elif time_freq == 'M1':
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        start_year = int(start_time[:4:])

        if int(start_year) < 2015:
            file_1 = os.path.join(data_dir, time_scale, insti, 'historical', var + '.csv')
            data_1 = read_xlsx_data(file_1, station_id)
            data_m1 = data_1.loc[start_time[:4:] + '-' + start_time[4::]:'2014']
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m2 = data_2.loc['2015':end_time[:4:] + '-' + end_time[4::]]
            data_m3 = pd.concat([data_m1, data_m2], axis=0)
        else:
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m3 = data_2.loc[start_time[:4:] + '-' + start_time[4::]:end_time[:4:] + '-' + end_time[4::]]

    elif time_freq == 'D1':
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        start_year = int(start_time[:4:])

        if int(start_year) < 2015:
            file_1 = os.path.join(data_dir, time_scale, insti, 'historical', var + '.csv')
            data_1 = read_xlsx_data(file_1, station_id)
            data_m1 = data_1.loc[start_time[:4:] + '-' + start_time[4:6:] + '-' + start_time[6::]:'2014']
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m2 = data_2.loc['2015':end_time[:4:] + '-' + end_time[4:6:] + '-' + end_time[6::]]
            data_m3 = pd.concat([data_m1, data_m2], axis=0)
        else:
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_m3 = data_2.loc[start_time[:4:] + '-' + start_time[4:6:] + '-' + start_time[6::]:end_time[:4:] + '-' + end_time[4::] + '-' + end_time[6::]]

    elif time_freq == 'D2':

        def read_and_merge_data(data_1, data_2, start_time, end_time, year):
            start_date = f"{year}-{start_time[:2]}-{start_time[2:]}"
            end_date = f"{year}-{end_time[:2]}-{end_time[2:]}"
            if year < 2014:
                return data_1.loc[start_date:end_date]
            else:
                return data_2.loc[start_date:end_date]

        years = stats_times[0]
        start_year = int(years.split(',')[0])
        end_year = int(years.split(',')[1])
        date_time = stats_times[1]
        start_time = date_time.split(',')[0]
        end_time = date_time.split(',')[1]

        if int(start_year) < 2015:
            file_1 = os.path.join(data_dir, time_scale, insti, 'historical', var + '.csv')
            data_1 = read_xlsx_data(file_1, station_id)
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
        else:
            file_2 = os.path.join(data_dir, time_scale, insti, scene, var + '.csv')
            data_2 = read_xlsx_data(file_2, station_id)
            data_1 = []

        data_m3 = pd.concat([read_and_merge_data(data_1, data_2, start_time, end_time, i) for i in range(start_year, end_year + 1)], axis=0)

    return data_m3


def create_datetimeindex(time_freq, time_info):
    '''
    根据传入的参考时段或预估时间，生成相应的datetimeindex
    '''
    # 首先筛选时间
    if time_freq == 'Y':
        s = time_info.split(',')[0]
        e = time_info.split(',')[1]
        e = str(int(e) + 1)
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # 'Y'

    elif time_freq in ['Q', 'M2']:
        s = time_info[0].split(',')[0]
        e = time_info[0].split(',')[1]
        mon_list = [int(val) for val in time_info[1].split(',')]
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # 'Q' or 'M2'
        time_index = time_index[time_index.month.isin(mon_list)]

    elif time_freq == 'M1':
        s = time_info.split(',')[0]
        e = time_info.split(',')[1]
        s = pd.to_datetime(s, format='%Y%m')
        e = pd.to_datetime(e, format='%Y%m') + pd.DateOffset(months=1)
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # M1

    elif time_freq == 'D1':
        s = time_info.split(',')[0]
        e = time_info.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D')  # D1

    elif time_freq == 'D2':  # ['%Y,%Y','%m%d,%m%d']
        s = time_info[0].split(',')[0]
        e = time_info[1].split(',')[1]
        s_mon = time_info[1].split(',')[0][:2]
        e_mon = time_info[1].split(',')[1][:2]
        s_day = time_info[1].split(',')[0][2:]
        e_day = time_info[1].split(',')[1][2:]
        dates = pd.date_range(start=s, end=e, freq='D')
        time_index = dates[((dates.month == s_mon) & (dates.day >= s_day)) | ((dates.month > s_mon) & (dates.month < e_mon)) | ((dates.month == e_mon) & (dates.day <= e_day))]

    time_index = time_index[~((time_index.month == 2) & (time_index.day == 29))]  # 由于数据原因，删除2月29号
    time_index_1 = time_index[(time_index.year >= 2020) & (time_index.year <= 2039)]  # 1.5 2020~2039 ssp126
    time_index_2 = time_index[(time_index.year >= 2040) & (time_index.year <= 2059)]  # 2.0 2040~2059 ssp245

    return time_index, time_index_1, time_index_2


def data_time_filter(nc_dict, time_index):
    '''
    根据上一个函数输出的time_index，筛选nc的time
    '''
    for _, sub_dict1 in nc_dict.items():  # evaluate_cmip[exp][insti][var]
        for _, sub_dict2 in sub_dict1.items():
            for key, df_data in sub_dict2.items():
                selected_data = df_data[df_data.index.isin(time_index)]
                sub_dict2[key] = selected_data

    return nc_dict


def get_station_info(sta_ids):
    '''
    获取站号信息
    '''
    # station_df = pd.DataFrame()
    # station_df['站号'] = [
    #     51886, 51991, 52602, 52633, 52645, 52657, 52707, 52713, 52737, 52745, 52754, 52765, 52818, 52825, 52833, 52836, 52842, 52851, 52853, 52855, 52856, 52859, 52862, 52863, 52866, 52868, 52869, 52874, 52875, 52876, 52877, 52908, 52942, 52943,
    #     52955, 52957, 52963, 52968, 52972, 52974, 56004, 56015, 56016, 56018, 56021, 56029, 56033, 56034, 56043, 56045, 56046, 56065, 56067, 56125, 56151
    # ]
    # station_df['站名'] = [
    #     '茫崖', '那陵格勒', '冷湖', '托勒', '野牛沟', '祁连', '小灶火', '大柴旦', '德令哈', '天峻', '刚察', '门源', '格尔木', '诺木洪', '乌兰', '都兰', '茶卡', '江西沟', '海晏', '湟源', '共和', '瓦里关', '大通', '互助', '西宁', '贵德', '湟中', '乐都', '平安', '民和', '化隆', '五道梁', '河卡', '兴海', '贵南', '同德', '尖扎', '泽库',
    #     '循化', '同仁', '沱沱河', '曲麻河', '治多', '杂多', '曲麻莱', '玉树', '玛多', '清水河', '玛沁', '甘德', '达日', '河南', '久治', '囊谦', '班玛'
    # ]
    
    path = cfg.FILES.STATION
    station_df = pd.read_csv(path, encoding='gbk')
    station_df.dropna(inplace=True)
    station_df = station_df[['区站号','站点名','经度','纬度']]
    station_df.columns = ['站号','站名','经度','纬度']
    new_station = station_df[station_df['站号'].isin(sta_ids)]
    
    return new_station


def choose_mod_path(inpath, data_source, insti, var, time_scale, yr, expri_i, res=None):
    # cmip数据路径选择
    """
    旧的 读取模式nc文件
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

    if data_source == 'original':
        # path = os.path.join(inpath, data_cource,path1,insti ,expri,var,filen)
        path_dir = os.path.join(inpath, data_source, path1, insti, expri, var)
        path = glob.glob(os.path.join(path_dir, f'{var}*{str(yr)}0101*.nc'))[0]
    else:
        # path = os.path.join(inpath, data_cource,res,path1,insti ,expri,var,filen)
        path_dir = os.path.join(inpath, data_source, res, path1, insti, expri, var)
        path = glob.glob(os.path.join(path_dir, f'{var}*{str(yr)}0101*.nc'))[0]

    return path


# inpath = r'C:\Users\MJY\Desktop\qhbh\zipdata\cmip6' # cmip6路径
# data_source = 'original'
# insti = 'BCC-CSM2-MR'
# var = 'pr'
# time_scale = 'daily'
# expri_i = 'ssp126'
# yr = 2018
# path = choose_mod_path(inpath, data_source, insti, var, time_scale, yr, expri_i, res=None)

# 确定年份
# if time_freq == 'Y':  # '%Y,%Y'
#     start_year = int(evaluate_times.split(',')[0])
#     end_year = int(evaluate_times.split(',')[1])

# elif time_freq in ['Q', 'M2', 'D2']:  # ['%Y,%Y','3,4,5']
#     years = evaluate_times[0]
#     start_year = int(years.split(',')[0])
#     end_year = int(years.split(',')[1])

# elif time_freq in ['M1', 'D1']:  # '%Y%m,%Y%m'
#     start_year = int(evaluate_times.split(',')[0][:4])
#     end_year = int(evaluate_times.split(',')[1][:4])

# 读取数据 生成dict
# inpath = r'C:\Users\MJY\Desktop\qhbh\zipdata\cmip6'

# evaluate_cmip = dict()
# for exp in ['ssp126', 'ssp245']:
#     evaluate_cmip[exp] = dict()
#     for insti in cmip_model:
#         evaluate_cmip[exp][insti] = dict()
#         tmp_lst = []
#         for year in range(start_year, end_year + 1):
#             tem_file_path = choose_mod_path(inpath=inpath, data_source=cmip_type, insti=insti, var=var, time_scale='daily', yr=year, expri_i=exp, res=cmip_res)

#             ds_tmp = xr.open_dataset(tem_file_path)
#             tmp_lst.append(ds_tmp)

#         tmp_all = xr.concat(tmp_lst, dim='time')
#         try:
#             tmp_all['time'] = tmp_all.indexes['time'].to_datetimeindex().normalize()
#         except:
#             tmp_all['time'] = tmp_all.indexes['time'].normalize()
#         evaluate_cmip[exp][insti][var] = tmp_all

if __name__ == '__main__':

    data_dir = r'C:\Users\MJY\Desktop\excel_data'

    time_scale = 'daily'
    insti = 'Set'
    scene = 'ssp126'
    var = 'tas'
    station_id = ['51886', '52602', '52633', '52645', '52657', '52707', '52713']

    # stats_times='2030,2040'
    # time_freq= 'Y'

    # stats_times=['2011,2040','12,1,2']
    # time_freq= 'Q'

    # stats_times='201102,202005'
    # time_freq= 'M1'

    # stats_times='20110205,20200505'
    # time_freq= 'D1'

    # stats_times=['2011,2040','0505,0805']
    # time_freq= 'D2'

    time_freq = 'M2'
    stats_times = ["2010,2025", "1,2"]

    new = read_model_data(data_dir, time_scale, insti, scene, var, stats_times, time_freq, station_id)

    time = new.index
    location = new.columns.tolist()
    da = xr.DataArray(new.values, coords=[time, location], dims=['time', 'location'])
    ds = xr.Dataset({'tas': da.astype('float32')})
