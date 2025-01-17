import os
import uuid
import numpy as np
import pandas as pd
import xarray as xr
import psycopg2
import copy
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
from Module02.page_water.wrapped.func02_result_stats import stats_result_4
from Utils.read_model_data import read_model_data, create_datetimeindex, data_time_filter, get_station_info
from Utils.config import cfg


def convert_nested_df(data):
    if isinstance(data, dict):
        return {k: convert_nested_df(v) for k, v in data.items()}
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    elif isinstance(data, pd.Series):
        return data.to_frame().T.round(1).to_dict(orient='records')
    else:
        return data
    

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
    
    # ------------------------------------------------------------------
    # 2.参数处理
    uuid4 = uuid.uuid4().hex
    data_out = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_out):
        os.makedirs(data_out)
        os.chmod(data_out, 0o007 | 0o070 | 0o700)
    
    if isinstance(cmip_model, str):
        cmip_model = cmip_model.split(',')

    if '集合' in cmip_model:
        cmip_model.remove('集合')
        cmip_model.append('Set')
        
    # ------------------------------------------------------------------
    # 3.从csv文件里面获取模式数据
    station_id = sta_ids.split(',')
    time_scale = 'daily'

    if os.name == 'nt':
        data_dir = r'C:\Users\MJY\Desktop\qhbh\zipdata\station_data\csv'  # 本地
    else:
        if cmip_type == 'original':
            data_dir = '/model_data/station_data/csv'
        elif cmip_type == 'delta':
            res_d = dict()
            res_d['25'] = '0.25deg'
            res_d['50'] = '0.52deg'
            res_d['100'] = '1.00deg'
            data_dir = '/model_data/station_data_delta/csv'
            data_dir = os.path.join(data_dir, res_d[cmip_res])

    # 循环读取
    # 下载参考时段和预估时段的模式数据
    refer_cmip = dict()
    evaluate_cmip = dict()
    for exp in ['ssp126', 'ssp245', 'ssp585']:
        refer_cmip[exp] = dict()
        evaluate_cmip[exp] = dict()

        for insti in cmip_model:
            refer_cmip[exp][insti] = dict()
            evaluate_cmip[exp][insti] = dict()

            for var in ['tas', 'pr']:
                # 根据参考时间段读取模式数据
                df = read_model_data(data_dir, time_scale, insti, exp, var, refer_years, time_freq, station_id)
                df = df.astype(float)
                df = df.mean(axis=1) # 多站求平均，代表水文站
                refer_cmip[exp][insti][var] = df

                # 根据预估时段读取模式数据
                df = read_model_data(data_dir,time_scale,insti,exp,var,evaluate_times,time_freq,station_id)
                df = df.astype(float)
                df = df.mean(axis=1) # 多站求平均，代表水文站
                evaluate_cmip[exp][insti][var] = df
    
    # ------------------------------------------------------------------
    # 4.根据预估时段，获取datetimeindex，然后进行filter
    time_index_e, time_index_15deg, time_index_20deg = create_datetimeindex(time_freq, evaluate_times)
    evaluate_cmip = data_time_filter(evaluate_cmip, time_index_e)  # 所有的数据

    # ------------------------------------------------------------------
    # 5.开始计算
    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['表格预估'] = dict()
    result_dict['时序图'] = dict()

    # 5.1 参考时段数据，计算基准期
    refer_result = dict()
    for exp, sub_dict1 in refer_cmip.items():
        refer_result[exp] = dict()
        for insti, data in sub_dict1.items():
            tem_daily = data['tas']
            pre_daily = data['pr']

            # 计算蒸发
            tem_monthly = tem_daily.resample('1M').mean()
            pre_monthly = pre_daily.resample('1M').sum()
            evp_monthly = 3100*tem_monthly/(3100+1.8*(pre_monthly**2)*np.exp((-34.4*tem_monthly)/(235+tem_monthly)))
            evp_monthly = evp_monthly.where(evp_monthly>0,0)

            # hbv-input
            date_time = tem_daily.index
            month = np.array(tem_daily.index.month)
            temp = tem_daily.values  # 气温 单位：度
            precip = pre_daily.values # 单位：mm
            q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
            base_p = q_sim.mean(axis=0).round(1)
            refer_result[exp][insti] = base_p

    # 5.预估-单情景-单模式
    single_cmip_res = dict()
    for exp, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti]['tmp']
        single_cmip_res[exp] = dict()
        for insti, sub_dict2 in sub_dict1.items():
            tem_daily = sub_dict2['tas']
            pre_daily = sub_dict2['pr']
        
            # 计算蒸发
            tem_monthly = tem_daily.resample('1M').mean()
            pre_monthly = pre_daily.resample('1M').sum()
            evp_monthly = 3100*tem_monthly/(3100+1.8*(pre_monthly**2)*np.exp((-34.4*tem_monthly)/(235+tem_monthly)))
            evp_monthly = evp_monthly.where(evp_monthly>0,0)
            
            # hbv-input
            date_time = tem_daily.index
            month = np.array(tem_daily.index.month)
            temp = tem_daily.values  # 气温 单位：度
            precip = pre_daily.values # 单位：mm
            q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
            q_sim = pd.DataFrame(q_sim, index=tem_daily.index, columns=['Q'])
            q_sim_yearly = q_sim.resample('1A').mean().round(1)
            single_cmip_res[exp][insti] = q_sim_yearly
    
    # 新增1.5和2.0 degree
    if len(time_index_15deg) != 0:
        print('deg1.5')
        degree15 = copy.deepcopy(evaluate_cmip)
        degree15 = data_time_filter(degree15, time_index_15deg)
        for exp, sub_dict1 in degree15.items():
            if exp == 'ssp126':
                single_cmip_res['1.5℃'] = dict()
                for insti, sub_dict2 in sub_dict1.items():
                    tem_daily = sub_dict2['tas']
                    pre_daily = sub_dict2['pr']
                
                    # 计算蒸发
                    tem_monthly = tem_daily.resample('1M').mean()
                    pre_monthly = pre_daily.resample('1M').sum()
                    evp_monthly = 3100*tem_monthly/(3100+1.8*(pre_monthly**2)*np.exp((-34.4*tem_monthly)/(235+tem_monthly)))
                    evp_monthly = evp_monthly.where(evp_monthly>0,0)
                    
                    # hbv-input
                    date_time = tem_daily.index
                    month = np.array(tem_daily.index.month)
                    temp = tem_daily.values  # 气温 单位：度
                    precip = pre_daily.values # 单位：mm
                    q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
                    q_sim = pd.DataFrame(q_sim, index=tem_daily.index, columns=['Q'])
                    q_sim_yearly = q_sim.resample('1A').mean().round(1)
                    single_cmip_res['1.5℃'][insti] = q_sim_yearly
    
    if len(time_index_20deg) != 0:
        print('deg2.0')
        degree20 = copy.deepcopy(evaluate_cmip)
        degree20 = data_time_filter(degree20, time_index_20deg)
        for exp, sub_dict1 in degree20.items():
            if exp == 'ssp245':
                single_cmip_res['2.0℃'] = dict()
                for insti, sub_dict2 in sub_dict1.items():
                    tem_daily = sub_dict2['tas']
                    pre_daily = sub_dict2['pr']
                
                    # 计算蒸发
                    tem_monthly = tem_daily.resample('1M').mean()
                    pre_monthly = pre_daily.resample('1M').sum()
                    evp_monthly = 3100*tem_monthly/(3100+1.8*(pre_monthly**2)*np.exp((-34.4*tem_monthly)/(235+tem_monthly)))
                    evp_monthly = evp_monthly.where(evp_monthly>0,0)
                    
                    # hbv-input
                    date_time = tem_daily.index
                    month = np.array(tem_daily.index.month)
                    temp = tem_daily.values  # 气温 单位：度
                    precip = pre_daily.values # 单位：mm
                    q_sim = hbv_main(len(temp), date_time, month, temp, precip, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca)
                    q_sim = pd.DataFrame(q_sim, index=tem_daily.index, columns=['Q'])
                    q_sim_yearly = q_sim.resample('1A').mean().round(1)
                    single_cmip_res['2.0℃'][insti] = q_sim_yearly

    single_cmip_res = stats_result_4(single_cmip_res, base_p, '唐乃亥', hydro_ids)
    
    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['表格历史'] = dict()
    result_dict['表格预估'] = dict()
    result_dict['时序图'] = dict()
    result_dict['表格历史']['观测'] = None
    result_dict['表格历史']['模拟观测'] = None
    result_dict['表格历史']['模拟模式'] = None
    result_dict['表格预估']['单模式'] = single_cmip_res
    result_dict['时序图']['基准期'] = refer_result

    # 最后遍历dict，如果是df就to_dict()
    result_dict = convert_nested_df(result_dict)

    return result_dict



if __name__ == '__main__':
    data_json = dict()
    data_json['time_freq'] = 'Y'
    data_json['evaluate_times'] = "2030,2060" # 预估时段时间条
    data_json['refer_years'] = '1985,2014'# 参考时段时间条
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
    result_dict = hbv_single_calc(data_json)