import os
import uuid
import time
import pandas as pd
import xarray as xr
import psycopg2
from io import StringIO
from psycopg2 import sql
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.data_processing import data_processing

# 其他行业


def other_features_stats(data_json):
    '''
    获取天擎数据，参数说明
    :param element：对应原型，传入的要素名称
        社会经济 gdp
        人口 pop
        能源生产总量 energy_production
        能源消费总量 energy_consumption
        5度采暖度日总量 heating_drgree_5
        18度采暖度日总量 heating_drgree_18
        旅客运输量 passenger_transport
        货物运输量 cargo_transport
        公路里程 road_mile
        WorldPOP-GDP gdp_worldpop
        WorldPOP-人口 pop_worldpop

    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'

    :param stats_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'

    :param sta_ids: 统计区域传入的站点，多站，传：'52866,52713,52714'

    :param interp_method: 对应原型的插值方法
        传参：
        克里金 - 'kriging'
        泛克里金 - 'uni_kriging'
        反距离权重 - 'idw'

    :param shp_path: shp文件
    :param output_filepath: 输出结果路径

    '''
    # 1.参数读取
    element = data_json['element']
    stats_times = data_json['stats_times']
    sta_ids = data_json.get('sta_ids')
    shp_path = data_json.get('shp_path')

    # 2.参数处理
    if shp_path is not None:
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径
        
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)    

    # 开始计算
    start_year = int(stats_times.split(',')[0])
    end_year = int(stats_times.split(',')[1])

    result_dict = dict()
    result_dict['uuid'] = uuid4
    
    if element == 'gdp':
        path = cfg.FILES.FILE01
        if sta_ids == '630000':
            qh_df = pd.read_excel(path, sheet_name='青海省国民经济生产总值', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
        else:
            sta_ids = sta_ids.split(',')
            station_info = pd.read_excel(path, sheet_name='Sheet1', header=None)
            station_info.columns = ['市县', '站号', '数值']
            station_info['站号'] = station_info['站号'].map(str)
            stations = station_info[station_info['站号'].isin(sta_ids)]['市县'].to_list()
            qh_df = pd.read_excel(path, sheet_name='地区生产总值', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
            qh_df = qh_df[['年份', '指标', '单位'] + stations]
    
    elif element == 'pop':
        path = cfg.FILES.FILE02
        if sta_ids == '630000':
            qh_df = pd.read_excel(path, sheet_name='人口（全省）', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
        else:
            sta_ids = sta_ids.split(',')
            station_info = pd.read_excel(path, sheet_name='Sheet1', header=None)
            station_info.columns = ['市县', '站号', '数值1', '数值2']
            station_info['站号'] = station_info['站号'].map(int).map(str)
            stations = station_info[station_info['站号'].isin(sta_ids)]['市县'].to_list()
            qh_df = pd.read_excel(path, sheet_name='人口（分县）', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
            qh_df = qh_df[['年份', '指标', '单位'] + stations]
    
    elif element == 'energy_production':
        path = cfg.FILES.FILE03
        qh_df = pd.read_excel(path, sheet_name='一次能源生产总量（万吨标准煤）', header=0)
        qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
    
    elif element == 'energy_consumption':
        path = cfg.FILES.FILE03
        qh_df = pd.read_excel(path, sheet_name='一次能源消费总量（万吨标准煤）', header=0)
        qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
    
    elif element == 'heating_drgree_5':
        path = cfg.FILES.FILE03
        if sta_ids == '630000':
            qh_df = pd.read_excel(path, sheet_name='5度采暖度日总量（℃•d）', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
        else:
            sta_ids = sta_ids.split(',')
            station_info = pd.read_excel(path, sheet_name='Sheet1', header=0)
            station_info['站号'] = station_info['站号'].map(int).map(str)
            stations = station_info[station_info['站号'].isin(sta_ids)]['站名'].to_list()
            qh_df = pd.read_excel(path, sheet_name='5度采暖度日总量（℃•d）', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
            qh_df = qh_df[['年份'] + stations]
    
    elif element == 'heating_drgree_18':
        path = cfg.FILES.FILE03
        if sta_ids == '630000':
            qh_df = pd.read_excel(path, sheet_name='18度采暖度日总量（℃•d）', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
        else:
            sta_ids = sta_ids.split(',')
            station_info = pd.read_excel(path, sheet_name='Sheet1', header=0)
            station_info['站号'] = station_info['站号'].map(int).map(str)
            stations = station_info[station_info['站号'].isin(sta_ids)]['站名'].to_list()
            qh_df = pd.read_excel(path, sheet_name='18度采暖度日总量（℃•d）', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
            qh_df = qh_df[['年份'] + stations]
    
    elif element == 'passenger_transport':
        path = cfg.FILES.FILE04
        qh_df = pd.read_excel(path, sheet_name='旅客运输量', header=0)
        qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
    
    elif element == 'cargo_transport':
        qh_df = pd.read_excel(path, sheet_name='货物运输量', header=0)
        qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
    
    elif element == 'road_mile':
        path = cfg.FILES.FILE04
        if sta_ids == '630000':
            qh_df = pd.read_excel(path, sheet_name='公路里程', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
        else:
            sta_ids = sta_ids.split(',')
            station_info = pd.read_excel(path, sheet_name='空间分布', header=0)
            station_info['站号'] = station_info['站号'].map(int).map(str)
            stations = station_info[station_info['站号'].isin(sta_ids)]['站名'].to_list()
            qh_df = pd.read_excel(path, sheet_name='公路里程', header=0)
            qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
            qh_df = qh_df[['年份','指标','单位'] + stations]
    
    result_dict['表格'] = qh_df.to_dict(orient='records')

    return result_dict


if __name__ == '__main__':
    # t1 = time.time()
    # t2 = time.time()
    # print(t2 - t1)

    # path = cfg.FILES.FILE04
    
    # years = '1980,2010'
    # start_year = int(years.split(',')[0])
    # end_year = int(years.split(',')[1])
    
    # sta_ids = '52869,52855,52875,52876,52874,52863,52877,52972,52765,52657,52853,52754,52856'
    
    # if sta_ids == '630000':
    #     qh_df = pd.read_excel(path, sheet_name='公路里程', header=0)
    #     qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
    # else:
    #     sta_ids = sta_ids.split(',')
    #     station_info = pd.read_excel(path, sheet_name='空间分布', header=0)
    #     station_info['站号'] = station_info['站号'].map(int).map(str)
    #     stations = station_info[station_info['站号'].isin(sta_ids)]['站名'].to_list()
    #     qh_df = pd.read_excel(path, sheet_name='公路里程', header=0)
    #     qh_df = qh_df[(qh_df['年份'] >= start_year) & (qh_df['年份'] <= end_year)]
    #     qh_df = qh_df[['年份','指标','单位'] + stations]
    pass
