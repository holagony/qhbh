import os
import uuid
import copy
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from Module02.page_risk.wrapped.drought_single import drought_cmip_single
from Module02.page_traffic.wrapped.func03_plot import interp_and_mask
from Module03.wrapped.plot_new import plot_and_save
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


def drought_esti(data_json):
    '''
    通过降水、气温、风速要素，计算交通不利日数

    :param element:
        干旱 drought

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
    
    :param refer_years: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'

    :param sta_ids: 传入的气象站点
    '''
    # 1.参数读取
    element = data_json['element']
    time_freq = data_json['time_freq']  # 控制预估时段
    evaluate_times = data_json['evaluate_times']  # 预估时段时间条
    refer_years = data_json['refer_years']  # 参考时段时间条
    sta_ids = data_json['sta_ids']  # 气象站 '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    cmip_type = data_json['cmip_type']  # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    cmip_res = data_json.get('cmip_res')  # 分辨率 1/5/10/25/50/100 km
    cmip_model = data_json['cmip_model']  # 模式，列表：['CanESM5','CESM2']等
    plot = data_json['plot']
    shp_path = data_json['shp_path']

    # ------------------------------------------------------------------
    # 2.参数处理
    uuid4 = uuid.uuid4().hex
    save_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.chmod(save_dir, 0o007 | 0o070 | 0o700)

    if isinstance(cmip_model, str):
        cmip_model = cmip_model.split(',')

    if os.name != 'nt':
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径
        method = 'idw'
    else:
        method = 'kri'

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
            data_dir = '/model_data/station_data/csv'  # 容器内
        elif cmip_type == 'delta':
            res_d = dict()
            res_d['25'] = '0.25deg'
            res_d['50'] = '0.50deg'
            res_d['100'] = '1deg'
            data_dir = '/model_data/station_data_delta/csv'  # 容器内
            data_dir = os.path.join(data_dir, res_d[cmip_res])

    # 验证期
    evaluate_cmip = dict()
    for exp in ['ssp126', 'ssp245', 'ssp585']:
        evaluate_cmip[exp] = dict()
        for insti in cmip_model:
            evaluate_cmip[exp][insti] = dict()
            for var in ['light_drought', 'medium_drought', 'heavy_drought', 'severe_drought']:
                df = read_model_data(data_dir, time_scale, insti, exp, var, evaluate_times, time_freq, station_id)
                df = df.astype(float)
                evaluate_cmip[exp][insti][var] = df

    # TODO 计算基准期需要获取tas和pr实时计算
    # refer_cmip = dict()
    # for exp in ['ssp126','ssp245','ssp585']:
    #     refer_cmip[exp] = dict()
    #     for insti in cmip_model:
    #         refer_cmip[exp][insti] = dict()
    #         for var in ['tas','pr']:
    #             df = read_model_data(data_dir,time_scale,insti,exp,var,evaluate_times,time_freq,station_id)
    #             df = df.astype(float)
    #             refer_cmip[exp][insti][var] = df

    # refer_cmip = drought_change_processing(refer_cmip)

    # 暂时先输出空
    refer_cmip = dict()
    for exp in ['ssp126', 'ssp245', 'ssp585']:
        refer_cmip[exp] = dict()
        for insti in cmip_model:
            refer_cmip[exp][insti] = None

    # ------------------------------------------------------------------
    # 4.根据预估时段，获取datetimeindex，然后进行filter
    time_index_e, time_index_15deg, time_index_20deg = create_datetimeindex(time_freq, evaluate_times)
    evaluate_cmip = data_time_filter(evaluate_cmip, time_index_e)  # 所有的数据

    # ------------------------------------------------------------------
    # 5.开始计算
    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['表格'] = dict()
    result_dict['时序图'] = dict()
    result_dict['分布图'] = dict()

    # 5.1 首先获取站号对应的站名，以及经纬度
    station_info = get_station_info(station_id)
    lon_list = station_info['经度'].tolist()
    lat_list = station_info['纬度'].tolist()
    sta_info = station_info[['站号', '站名']]
    result_dict['站号'] = sta_info.to_dict(orient='records')

    # 5.2 承灾体静态数据插值到站点
    interp_lon = xr.DataArray(lon_list, dims="location", coords={
        "location": station_id,
    })
    interp_lat = xr.DataArray(lat_list, dims="location", coords={
        "location": station_id,
    })
    czt_path = cfg.FILES.DROUGHT_CZT
    czt_data = xr.open_dataset(czt_path)
    czt_data = czt_data.interp(lat=interp_lat, lon=interp_lon, method='nearest')

    # 5.3 孕灾环境静态数据插值到站点
    yz_path = cfg.FILES.DROUGHT_YZ
    yz_data = xr.open_dataset(yz_path)
    yz_data = yz_data.interp(lat=interp_lat, lon=interp_lon, method='nearest')

    # 5.4 GDP静态数据插值到站点
    gdp_path = cfg.FILES.DROUGHT_GDP
    gdp_data = xr.open_dataset(gdp_path)
    gdp_data = gdp_data.interp(lat=interp_lat, lon=interp_lon, method='nearest')

    # 5.5 表格-预估-各个情景的单模式
    single_cmip_res = dict()
    for exp, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti][var]
        single_cmip_res[exp] = dict()
        for insti, data in sub_dict1.items():
            res_table = drought_cmip_single(data, czt_data, yz_data, gdp_data)
            single_cmip_res[exp][insti] = res_table

    # 新增1.5和2.0 degree
    if len(time_index_15deg) != 0:
        degree15 = copy.deepcopy(evaluate_cmip)
        degree15 = data_time_filter(degree15, time_index_15deg)
        for exp, sub_dict1 in degree15.items():
            if exp == 'ssp126':
                single_cmip_res['1.5℃'] = dict()
                for insti, data in sub_dict1.items():
                    res_table = drought_cmip_single(data, czt_data, yz_data, gdp_data)
                    single_cmip_res['1.5℃'][insti] = res_table

    if len(time_index_20deg) != 0:
        degree20 = copy.deepcopy(evaluate_cmip)
        degree20 = data_time_filter(degree20, time_index_20deg)
        for exp, sub_dict1 in degree20.items():
            if exp == 'ssp245':
                single_cmip_res['2.0℃'] = dict()
                for insti, data in sub_dict1.items():
                    res_table = drought_cmip_single(data, czt_data, yz_data, gdp_data)
                    single_cmip_res['2.0℃'][insti] = res_table

    result_dict['表格']['预估单模式'] = single_cmip_res

    # 5.6 时序图-基准期
    result_dict['时序图']['基准期'] = refer_cmip

    # ------------------------------------------------------------------
    # 6 分布图
    if plot == 1:
        all_png = dict()
        for exp, sub_dict1 in single_cmip_res.items():
            if exp in ['ssp126', 'ssp245', 'ssp585']:
                all_png[exp] = dict()
                for insti, stats_table in sub_dict1.items():
                    all_png[exp][insti] = dict()
                    for i in tqdm(range(len(stats_table))):
                        # for i in tqdm(range(77,78)):
                        value_list = stats_table.iloc[i, 1:-3].tolist()
                        year_name = stats_table.iloc[i, 0]
                        exp_name = exp
                        insti_name = insti
                        bar_name = str(stats_table.iloc[i, 0])
                        # 插值/掩膜/画图/保存
                        mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                        png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, save_dir, '干旱风险指数' + bar_name)
    
                        # 转url
                        png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                        png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
    
                        all_png[exp][insti][year_name] = png_path
    else:
        all_png = None

    result_dict['分布图']['预估单模式'] = all_png

    # 最后遍历dict，如果是df就to_dict()
    result_dict = convert_nested_df(result_dict)

    return result_dict


if __name__ == '__main__':
    data_json = dict()
    data_json['time_freq'] = 'Y'
    data_json['evaluate_times'] = '2025,2100'  # 预估时段时间条
    data_json['refer_years'] = '1985,2014'  # 参考时段时间条
    data_json[
        'sta_ids'] = '51886,52602,52633,52645,52657,52707,52713,52737,52745,52754,52765,52818,52825,52833,52836,52842,52853,52855,52856,52862,52863,52866,52868,52869,52874,52876,52877,52908,52943,52955,52957,52963,52968,52972,52974,56004,56016,56018,56021,56029,56033,56034,56043,56045,56046,56065,56067,56125,56151'
    data_json['cmip_type'] = 'original'  # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None  # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ['KIOST-ESM', 'MPI-ESM1-2-LR']  # 模式，列表：['CanESM5','CESM2']等
    data_json['plot'] = 0
    data_json['shp_path'] = r'C:\Users\MJY\Desktop\qh_hx\qh_hx\qh_hx.shp'
    data_json['element'] = 'drought'
    result_dict = drought_esti(data_json)
