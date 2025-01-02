import os
import uuid
import time
import glob
import numpy as np
import pandas as pd
import xarray as xr
import psycopg2
from tqdm import tqdm
from io import StringIO
from psycopg2 import sql
from datetime import date, datetime, timedelta
from Utils.config import cfg
from Utils.data_loader_with_threads import get_database_data
from Module02.page_risk.wrapped.station_processing import drought_change_processing
from Module02.page_risk.wrapped.func01_table_stats import table_stats_rain
from Module02.page_risk.wrapped.drought_multi import drought_cmip_multi
from Module02.page_risk.wrapped.drought_single import drought_cmip_single
from Module02.page_traffic.wrapped.func_plot import interp_and_mask#, plot_and_save
from Utils.read_model_data import read_model_data
from Module02.page_risk.wrapped.mci import calc_mci
from Module03.wrapped.plot_new import plot_and_save

# 气候变化风险预估--干旱


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

    # 2.参数处理
    uuid4 = uuid.uuid4().hex
    save_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.chmod(save_dir, 0o007 | 0o070 | 0o700)

    if os.name != 'nt':
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径
        method = 'idw'
    else:
        method = 'kri'
        
    if '集合' in cmip_model:
        cmip_model.remove('集合')
        cmip_model.append('Set')

    ######################################################
    # 站点数据获取从数据库获取
    table_name = 'qh_qhbh_calc_elements_day'
    element_str = 'light_drought,medium_drought,heavy_drought,severe_drought'
    sta_ids = tuple(sta_ids.split(','))
    refer_df = get_database_data(sta_ids, element_str, table_name, time_freq, refer_years)

    ######################################################
    # 读取模式csv数据
    res_d = dict()
    res_d['25'] = '0.25deg'
    res_d['50'] = '0.52deg'
    res_d['100'] = '1deg'

    if os.name == 'nt':
        data_dir = r'C:\Users\MJY\Desktop\qhbh\zipdata\station_data\csv'  # 本地
    else:
        if cmip_type == 'original':
            data_dir = '/model_data/station_data/csv'  # 容器内
        elif cmip_type == 'delta':
            data_dir = '/model_data/station_data_delta/csv'  # 容器内
            data_dir = os.path.join(data_dir, res_d[cmip_res])

    time_scale = 'daily'
    evaluate_cmip = dict()
    station_id = list(sta_ids)
    for exp in ['ssp126','ssp245','ssp585']:
    # for exp in ['ssp245']:
        evaluate_cmip[exp] = dict()
        for insti in cmip_model:
            evaluate_cmip[exp][insti] = dict()
            for var in ['light_drought','medium_drought','heavy_drought','severe_drought']:
                excel_data = read_model_data(data_dir, time_scale, insti, exp, var, evaluate_times, time_freq, station_id)
                # 转nc
                time_tmp = excel_data.index
                location_tmp = excel_data.columns.tolist()
                da = xr.DataArray(excel_data.values, coords=[time_tmp, location_tmp], dims=['time', 'location'])
                ds_excel = xr.Dataset({var: da.astype('float32')})
                evaluate_cmip[exp][insti][var] = ds_excel
                
    ######################################################
    # 重要!!! 获取站点经纬度
    df_unique = refer_df.drop_duplicates(subset='Station_Id_C')  # 删除重复行
    lon_list = df_unique['Lon'].tolist()
    lat_list = df_unique['Lat'].tolist()
    sta_list = df_unique['Station_Id_C'].tolist()

    ######################################################
    # 承灾体静态数据插值到站点
    interp_lon = xr.DataArray(lon_list, dims="location", coords={"location": sta_list,})
    interp_lat = xr.DataArray(lat_list, dims="location", coords={"location": sta_list,})
    czt_path = cfg.FILES.DROUGHT_CZT
    czt_data = xr.open_dataset(czt_path)
    czt_data = czt_data.interp(lat=interp_lat, lon=interp_lon, method='nearest')

    # 孕灾环境静态数据插值到站点
    yz_path = cfg.FILES.DROUGHT_YZ
    yz_data = xr.open_dataset(yz_path)
    yz_data = yz_data.interp(lat=interp_lat, lon=interp_lon, method='nearest')
    
    # GDP静态数据插值到站点
    gdp_path = cfg.FILES.DROUGHT_GDP
    gdp_data = xr.open_dataset(gdp_path)
    gdp_data = gdp_data.interp(lat=interp_lat, lon=interp_lon, method='nearest')
    
    ######################################################
    ##### 模式数据处理
    # 首先筛选时间
    if time_freq == 'Y':
        s = evaluate_times.split(',')[0]
        e = evaluate_times.split(',')[1]
        e = str(int(e) + 1)
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # 'Y'

    elif time_freq in ['Q', 'M2']:
        s = evaluate_times[0].split(',')[0]
        e = evaluate_times[1].split(',')[1]
        mon_list = [int(val) for val in evaluate_times[1].split(',')]
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # 'Q' or 'M2'
        time_index = time_index[time_index.month.isin(mon_list)]

    elif time_freq == 'M1':
        s = evaluate_times.split(',')[0]
        e = evaluate_times.split(',')[1]
        s = pd.to_datetime(s, format='%Y%m')
        e = pd.to_datetime(e, format='%Y%m') + pd.DateOffset(months=1)
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # M1

    elif time_freq == 'D1':
        s = evaluate_times.split(',')[0]
        e = evaluate_times.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D')  # D1

    elif time_freq == 'D2':  # ['%Y,%Y','%m%d,%m%d']
        s = evaluate_times[0].split(',')[0]
        e = evaluate_times[1].split(',')[1]
        s_mon = evaluate_times[1].split(',')[0][:2]
        e_mon = evaluate_times[1].split(',')[1][:2]
        s_day = evaluate_times[1].split(',')[0][2:]
        e_day = evaluate_times[1].split(',')[1][2:]
        dates = pd.date_range(start=s, end=e, freq='D')
        time_index = dates[((dates.month == s_mon) & (dates.day >= s_day)) | ((dates.month > s_mon) & (dates.month < e_mon)) | ((dates.month == e_mon) & (dates.day <= e_day))]

    time_index = time_index[~((time_index.month == 2) & (time_index.day == 29))]  # 由于数据原因，删除2月29号
    
    # 插值到多个站点
    interp_lon = xr.DataArray(lon_list, dims="location", coords={"location": sta_list,})
    interp_lat = xr.DataArray(lat_list, dims="location", coords={"location": sta_list,})

    for _, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti]['tmp']
        for _, sub_dict2 in sub_dict1.items():
            for key, ds_data in sub_dict2.items():
                try:
                    selected_data = ds_data.sel(time=time_index)
                except:
                    selected_data = ds_data
                # selected_data = selected_data.interp(lat=interp_lat, lon=interp_lon, method='nearest')
                sub_dict2[key] = selected_data
    
    ######################################################
    # 开始计算
    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['表格'] = dict()
    result_dict['时序图'] = dict()
    result_dict['分布图'] = dict()

    # 首先获取站号对应的站名
    station_df = pd.DataFrame()
    station_df['站号'] = [
        51886, 51991, 52602, 52633, 52645, 52657, 52707, 52713, 52737, 52745, 52754, 52765, 52818, 52825, 52833, 52836, 52842, 52851, 52853, 52855, 52856, 52859, 52862, 52863, 52866, 52868, 52869,
        52874, 52875, 52876, 52877, 52908, 52942, 52943, 52955, 52957, 52963, 52968, 52972, 52974, 56004, 56015, 56016, 56018, 56021, 56029, 56033, 56034, 56043, 56045, 56046, 56065, 56067, 56125,
        56151
    ]
    station_df['站名'] = [
        '茫崖', '那陵格勒', '冷湖', '托勒', '野牛沟', '祁连', '小灶火', '大柴旦', '德令哈', '天峻', '刚察', '门源', '格尔木', '诺木洪', '乌兰', '都兰', '茶卡', '江西沟', '海晏', '湟源', '共和', '瓦里关', '大通', '互助', '西宁', '贵德', '湟中', '乐都', '平安', '民和',
        '化隆', '五道梁', '河卡', '兴海', '贵南', '同德', '尖扎', '泽库', '循化', '同仁', '沱沱河', '曲麻河', '治多', '杂多', '曲麻莱', '玉树', '玛多', '清水河', '玛沁', '甘德', '达日', '河南', '久治', '囊谦', '班玛'
    ]
    station_df['站号'] = station_df['站号'].map(str)
    new_station = station_df[station_df['站号'].isin(sta_ids)]
    result_dict['站号'] = new_station.to_dict(orient='records')

    # 1.表格-历史
    # refer_df = drought_change_processing(refer_df) # 站点数据处理 对应历史
    # stats_result_his, _, _ = table_stats_rain(refer_df, disaster)
    # result_dict['表格']['历史'] = stats_result_his.to_dict(orient='records')

    # 2.表格-预估-各个情景的集合
    # evaluate_cmip_res = dict()
    # for exp, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti][var]
    #     evaluate_cmip_res[exp] = dict()
    #     for var in ['tas', 'pr']:
    #         ds_list = []
    #         for insti, sub_dict2 in sub_dict1.items():
    #             ds = sub_dict2[var]
    #             ds_list.append(ds)

    #         ds_daily = xr.concat(ds_list, 'new_dim')
    #         ds_daily = ds_daily.mean(dim='new_dim')
    #         evaluate_cmip_res[exp][var] = ds_daily  # 先平均情景下相同要素的xr
    
    # # 调用生成表格
    # res_table_multi = drought_cmip_multi(evaluate_cmip_res, czt_data, yz_data, gdp_data)
    # result_dict['表格']['预估集合'] = res_table_multi

    # 3.表格-预估-各个情景的单模式
    # evaluate_cmip 原始插值后数据
    single_cmip_res = drought_cmip_single(evaluate_cmip, czt_data, yz_data, gdp_data)
    result_dict['表格']['预估单模式'] = single_cmip_res
    
    # 4.时序图-各个情景的集合
    std_percent = dict()
    for exp, sub_dict in single_cmip_res.items():
        std_percent[exp] = dict()
        array_list = []
        for insti, res_df in sub_dict.items():
            res_df = pd.DataFrame(res_df)
            res_df.set_index('时间', inplace=True)
            array_list.append(res_df.iloc[:-4, :].values[None])
            array = np.concatenate(array_list, axis=0)
            std = np.std(array, ddof=1, axis=0).round(2)
            per25 = np.percentile(array, 25, axis=0).round(2)
            per75 = np.percentile(array, 75, axis=0).round(2)

            std = pd.DataFrame(std, index=res_df.index[:-4], columns=res_df.columns)
            per25 = pd.DataFrame(per25, index=res_df.index[:-4], columns=res_df.columns)
            per75 = pd.DataFrame(per75, index=res_df.index[:-4], columns=res_df.columns)

            std.reset_index(drop=False, inplace=True)
            per25.reset_index(drop=False, inplace=True)
            per75.reset_index(drop=False, inplace=True)

            std_percent[exp]['1倍标准差'] = std.to_dict(orient='records')
            std_percent[exp]['百分位数25'] = per25.to_dict(orient='records')
            std_percent[exp]['百分位数75'] = per75.to_dict(orient='records')

    result_dict['时序图'] = std_percent

    # 5.分布图 实时画（后面改为提取提前画好的图）
    if plot == 1:
        all_png = dict()
        for exp, sub_dict1 in single_cmip_res.items():
            all_png[exp] = dict()
            for insti, stats_table in sub_dict1.items():
                all_png[exp][insti] = dict()
                stats_table = pd.DataFrame(stats_table)
                
                for i in tqdm(range(len(stats_table))):
                # for i in tqdm(range(77,78)):
                    value_list = stats_table.iloc[i, 1:-3].tolist()
                    year_name = stats_table.iloc[i, 0]
                    exp_name = exp
                    insti_name = insti
                    bar_name = str(stats_table.iloc[i,0])
                    # 插值/掩膜/画图/保存
                    mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                    png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, save_dir, '干旱风险指数'+bar_name)
                    
                    # 转url
                    png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                    png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url

                    all_png[exp][insti][year_name] = png_path

        # 预估-集合数据画图
        # all_png1 = dict()
        # for exp, stats_table1 in res_table_multi.items():
        #     all_png1[exp] = dict()
        #     stats_table1 = pd.DataFrame(stats_table1)
        #     for i in tqdm(range(len(stats_table1))):
        #         value_list = stats_table1.iloc[i, 1:-3].tolist()
        #         year_name = stats_table1.iloc[i, 0]
        #         exp_name = exp
        #         insti_name = '集合'
        #         # 插值/掩膜/画图/保存
        #         mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
        #         png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, data_dir)

        #         # 转url
        #         png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
        #         png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        #         all_png1[exp][year_name] = png_path

        # 历史-观测画图
        # all_png2 = dict()
        # stats_result_his = pd.DataFrame(stats_result_his)
        # for i in tqdm(range(len(stats_result_his))):
        #     value_list = stats_result_his.iloc[i, 1:-3].tolist()
        #     year_name = stats_result_his.iloc[i, 0]
        #     exp_name = ''
        #     insti_name = ''
        #     # 插值/掩膜/画图/保存
        #     mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
        #     png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, data_dir)

        #     # 转url
        #     png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
        #     png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        #     all_png2[year_name] = png_path

    else:  # 直接获取现成的，目前没做，所有图片路径都是None
        all_png = dict()
        # all_png1 = dict()
        # all_png2 = dict()

    result_dict['分布图']['预估单模式'] = all_png
    # result_dict['分布图']['预估集合'] = all_png1
    # result_dict['分布图']['历史'] = all_png2

    return result_dict


if __name__ == '__main__':
    data_json = dict()
    data_json['time_freq'] = 'Y'
    data_json['evaluate_times'] = '2025,2100'  # 预估时段时间条
    data_json['refer_years'] = '2018,2024'  # 参考时段时间条
    data_json['sta_ids'] = '51886,52602,52633,52645,52657,52707,52713,52737,52745,52754,52765,52818,52825,52833,52836,52842,52853,52855,52856,52862,52863,52866,52868,52869,52874,52876,52877,52908,52943,52955,52957,52963,52968,52972,52974,56004,56016,56018,56021,56029,56033,56034,56043,56045,56046,56065,56067,56125,56151'
    data_json['cmip_type'] = 'original'  # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None  # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ['KIOST-ESM','MPI-ESM1-2-LR']  # 模式，列表：['CanESM5','CESM2']等
    data_json['plot'] = 1
    data_json['shp_path'] = r'C:\Users\MJY\Desktop\qh_hx\qh_hx\qh_hx.shp'
    data_json['element'] = 'drought'
    result_dict = drought_esti(data_json)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
