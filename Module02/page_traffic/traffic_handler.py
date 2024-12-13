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
from Module02.page_traffic.wrapped.traffic_processing import station_traffic_processing
from Module02.page_traffic.wrapped.func01_traffic_multi import traffic_cmip_multi
from Module02.page_traffic.wrapped.func02_traffic_single import traffic_cmip_single
from Module02.page_climate.wrapped.func01_table_stats import table_stats_simple
from Module02.page_traffic.wrapped.func03_plot import interp_and_mask, plot_and_save
from Utils.read_model_data import read_model_data

# 交通影响预估

def choose_mod_path(inpath, data_source, insti, var, time_scale, yr, expri_i, res=None):
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

def traffic_esti(data_json):
    '''
    通过降水、气温、风速要素，计算交通不利日数

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
    time_freq = data_json['time_freq'] # 控制预估时段
    evaluate_times = data_json['evaluate_times'] # 预估时段时间条
    refer_years = data_json['refer_years'] # 参考时段时间条
    sta_ids = data_json['sta_ids'] # 气象站 '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    cmip_type = data_json['cmip_type'] # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    cmip_res = data_json.get('cmip_res') # 分辨率 1/5/10/25/50/100 km
    cmip_model = data_json['cmip_model'] # 模式，列表：['CanESM5','CESM2']等
    plot = data_json['plot']
    # method = data_json['method']
    shp_path = data_json['shp_path']
    method = 'idw'

    # inpath = '/cmip_data'
    # inpath = r'C:\Users\MJY\Desktop\qhbh\zipdata' # cmip6路径
    if shp_path is not None:
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径
    
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
    # 站点数据获取
    table_name = 'qh_qhbh_cmadaas_day'
    element_str = 'PRE_Time_2020,TEM_Avg,WIN_S_2mi_Avg'
    
    # 从数据库获取
    sta_ids = tuple(sta_ids.split(','))
    refer_df = get_database_data(sta_ids, element_str, table_name, time_freq, refer_years)

    ######################################################
    # 模式数据获取
    # 先确定年份
    if time_freq == 'Y':  # '%Y,%Y'
        start_year = int(evaluate_times.split(',')[0])
        end_year = int(evaluate_times.split(',')[1])

    elif time_freq in ['Q', 'M2', 'D2']:  # ['%Y,%Y','3,4,5']
        years = evaluate_times[0]
        start_year = int(years.split(',')[0])
        end_year = int(years.split(',')[1])

    elif time_freq in ['M1', 'D1']:  # '%Y%m,%Y%m'
        start_year = int(evaluate_times.split(',')[0][:4])
        end_year = int(evaluate_times.split(',')[1][:4])

    # 读取数据 并且concat
    # evaluate_cmip = dict()
    # for exp in ['ssp126','ssp245']:
    #     evaluate_cmip[exp] = dict()
    #     for insti in cmip_model:
    #         evaluate_cmip[exp][insti] = dict()
    #         for var in ['tas', 'pr', 'uas', 'vas']:
    #             tmp_lst = []
    #             for year in range(start_year,end_year+1):
    #                 tem_file_path = choose_mod_path(inpath=inpath, 
    #                                                 data_source=cmip_type,
    #                                                 insti=insti, 
    #                                                 var=var, 
    #                                                 time_scale='daily', 
    #                                                 yr=year, 
    #                                                 expri_i=exp, 
    #                                                 res=cmip_res)
    
    #                 ds_tmp = xr.open_dataset(tem_file_path)
    #                 tmp_lst.append(ds_tmp)
                
    #             tmp_all = xr.concat(tmp_lst, dim='time')
    #             try:
    #                 tmp_all['time'] = tmp_all.indexes['time'].to_datetimeindex().normalize()
    #             except:
    #                 tmp_all['time'] = tmp_all.indexes['time'].normalize()
    #             evaluate_cmip[exp][insti][var] = tmp_all
    
    # 直接读取excel
    res_d = dict()
    res_d['25'] = '0.25deg'
    res_d['50'] = '0.52deg'
    res_d['100'] = '1deg'
    
    if os.name == 'nt':
        data_dir = r'C:\Users\MJY\Desktop\station_data\csv' # 本地
    else:
        if cmip_type == 'original':
            data_dir = '/model_data/station_data/csv' # 容器内
        elif cmip_type == 'delta':
            data_dir = '/model_data/station_data_delta/csv' # 容器内
            data_dir = os.path.join(data_dir, res_d[cmip_res])
    
    time_scale= 'daily'
    evaluate_cmip = dict()
    station_id = list(sta_ids)
    for exp in ['ssp126', 'ssp245', 'ssp585']:
        evaluate_cmip[exp] = dict()
        for insti in cmip_model:
            evaluate_cmip[exp][insti] = dict()
            for var in ['tas', 'pr', 'uas', 'vas']:
                excel_data = read_model_data(data_dir,time_scale,insti,exp,var,evaluate_times,time_freq,station_id)
                # 转nc
                time_tmp = excel_data.index
                location_tmp = excel_data.columns.tolist()
                da = xr.DataArray(excel_data.values, coords=[time_tmp, location_tmp], dims=['time', 'location'])
                ds_excel = xr.Dataset({var: da.astype('float32')})
                evaluate_cmip[exp][insti][var] = ds_excel

    ######################################################
    # 数据处理
    ##### 站点数据处理为交通不利日数
    refer_df = station_traffic_processing(refer_df, element_str)
    
    # 重要!!!
    df_unique = refer_df.drop_duplicates(subset='Station_Id_C') # 删除重复行
    lon_list = df_unique['Lon'].tolist()
    lat_list = df_unique['Lat'].tolist()
    sta_list = df_unique['Station_Id_C'].tolist()
    
    ######################################################
    ##### 模式数据处理
    # 首先筛选时间
    if time_freq == 'Y':
        s = evaluate_times.split(',')[0]
        e = evaluate_times.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D') # 'Y'

    elif time_freq in ['Q', 'M2']:
        s = evaluate_times[0].split(',')[0]
        e = evaluate_times[1].split(',')[1]
        mon_list = [int(val) for val in evaluate_times[1].split(',')]
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1]  # 'Q' or 'M2'
        time_index = time_index[time_index.month.isin(mon_list)]
    
    elif time_freq == 'M1':
        s = evaluate_times.split(',')[0]
        e = evaluate_times.split(',')[1]
        s = pd.to_datetime(s,format='%Y%m')
        e = pd.to_datetime(e,format='%Y%m') + pd.DateOffset(months=1)
        time_index = pd.date_range(start=s, end=e, freq='D')[:-1] # M1
    
    elif time_freq == 'D1':
        s = evaluate_times.split(',')[0]
        e = evaluate_times.split(',')[1]
        time_index = pd.date_range(start=s, end=e, freq='D') # D1
    
    elif time_freq == 'D2': # ['%Y,%Y','%m%d,%m%d']
        s = evaluate_times[0].split(',')[0]
        e = evaluate_times[1].split(',')[1]
        s_mon = evaluate_times[1].split(',')[0][:2]
        e_mon = evaluate_times[1].split(',')[1][:2]
        s_day = evaluate_times[1].split(',')[0][2:]
        e_day = evaluate_times[1].split(',')[1][2:]
        dates = pd.date_range(start=s, end=e, freq='D')
        time_index = dates[((dates.month==s_mon) & (dates.day>=s_day)) | ((dates.month>s_mon) & (dates.month<e_mon)) | ((dates.month==e_mon) & (dates.day<=e_day))]
        
    time_index = time_index[~((time_index.month==2) & (time_index.day==29))] # 由于数据原因，删除2月29号
    
    # 插值到多个站点
    interp_lon = xr.DataArray(lon_list, dims="location", coords={"location": sta_list,})
    interp_lat = xr.DataArray(lat_list, dims="location", coords={"location": sta_list,})
    
    for _, sub_dict1 in evaluate_cmip.items(): # evaluate_cmip[exp][insti]['tmp']
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
        51886, 51991, 52602, 52633, 52645, 52657, 52707, 52713, 52737, 52745, 52754, 52765, 52818, 52825, 52833, 52836, 52842, 52851, 52853, 52855, 52856, 
        52859, 52862, 52863, 52866, 52868, 52869, 52874, 52875, 52876, 52877, 52908, 52942, 52943, 52955, 52957, 52963, 52968, 52972, 52974, 56004, 56015, 
        56016, 56018, 56021, 56029, 56033, 56034, 56043, 56045, 56046, 56065, 56067, 56125, 56151]
    station_df['站名'] = [
        '茫崖', '那陵格勒', '冷湖', '托勒', '野牛沟', '祁连', '小灶火', '大柴旦', '德令哈', '天峻', '刚察', '门源', '格尔木', '诺木洪', '乌兰', '都兰', '茶卡', 
        '江西沟', '海晏', '湟源', '共和', '瓦里关', '大通', '互助', '西宁', '贵德', '湟中', '乐都', '平安', '民和', '化隆', '五道梁', '河卡', '兴海', '贵南', '同德',
        '尖扎', '泽库', '循化', '同仁', '沱沱河', '曲麻河', '治多', '杂多', '曲麻莱', '玉树', '玛多', '清水河', '玛沁', '甘德', '达日', '河南', '久治', '囊谦', '班玛']
    station_df['站号'] = station_df['站号'].map(str)
    new_station = station_df[station_df['站号'].isin(sta_ids)]
    result_dict['站号'] = new_station.to_dict(orient='records')

    # 1.表格-历史
    stats_result_his, _, _ = table_stats_simple(refer_df, 'traffic')
    result_dict['表格']['历史'] = stats_result_his.to_dict(orient='records')
    
    # 添加 基准期
    base_p=stats_result_his.iloc[0:-4,1::].mean().to_frame().T.reset_index(drop=True)

    
    # 2.表格-预估-各个情景的集合
    evaluate_cmip_res = dict()
    for exp, sub_dict1 in evaluate_cmip.items():  # evaluate_cmip[exp][insti]['tas']
        evaluate_cmip_res[exp] = dict()
        for var in ['tas', 'pr', 'uas', 'vas']:
            ds_list = []
            for insti, sub_dict2 in sub_dict1.items():            
                ds = sub_dict2[var]
                ds_list.append(ds)

            ds_daily = xr.concat(ds_list, 'new_dim')
            ds_daily = ds_daily.mean(dim='new_dim')
            evaluate_cmip_res[exp][var] = ds_daily # 先平均情景下相同要素的xr
            
    # 调用生成表格
    res_table_multi = traffic_cmip_multi(evaluate_cmip_res, stats_result_his)
    result_dict['表格']['预估集合'] = res_table_multi
        
    # 3.表格-预估-各个情景的单模式
    # evaluate_cmip 原始插值后数据
    single_cmip_res = traffic_cmip_single(evaluate_cmip, stats_result_his)                
    result_dict['表格']['预估单模式'] = single_cmip_res
                
    # 4.时序图-各个情景的集合
    std_percent = dict()
    for exp, sub_dict in single_cmip_res.items():
        std_percent[exp] = dict()
        array_list= []
        for insti, res_df in sub_dict.items():
            res_df = pd.DataFrame(res_df)
            res_df.set_index('时间',inplace=True)
            
            array_list.append(res_df.iloc[:-7,:].values[None])
            array = np.concatenate(array_list,axis=0)
            
            std = np.std(array, ddof=1, axis=0).round(2) # 只有一个模式的时候，std是nan
            per25 = np.percentile(array, 25, axis=0).round(2)
            per75 = np.percentile(array, 75, axis=0).round(2)
            
            std = pd.DataFrame(std, index=res_df.index[:-7], columns=res_df.columns)
            per25 = pd.DataFrame(per25, index=res_df.index[:-7], columns=res_df.columns)
            per75 = pd.DataFrame(per75, index=res_df.index[:-7], columns=res_df.columns)
            
            std.reset_index(drop=False,inplace=True)
            per25.reset_index(drop=False,inplace=True)
            per75.reset_index(drop=False,inplace=True)
            
            std_percent[exp]['1倍标准差'] = std.to_dict(orient='records')
            std_percent[exp]['百分位数25'] = per25.to_dict(orient='records')
            std_percent[exp]['百分位数75'] = per75.to_dict(orient='records')
    
    result_dict['时序图'] = std_percent
    result_dict['时序图']['基准期'] = base_p.to_dict(orient='records').copy()
    
    # 5.分布图 实时画（后面改为提取提前画好的图）
    if plot == 1:
        all_png = dict()
        for exp, sub_dict1 in single_cmip_res.items():
            all_png[exp] = dict()
            for insti,stats_table in sub_dict1.items():
                all_png[exp][insti] = dict()
                stats_table = pd.DataFrame(stats_table)
                for i in tqdm(range(len(stats_table))):
                    value_list = stats_table.iloc[i,1:-3].tolist()
                    year_name = stats_table.iloc[i,0]
                    exp_name = exp
                    insti_name = insti
                    # 插值/掩膜/画图/保存
                    mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                    png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, data_dir)
                    
                    # 转url
                    png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                    png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url

                    all_png[exp][insti][year_name] = png_path


        # 预估-集合数据画图
        all_png1 = dict()
        for exp, stats_table1 in res_table_multi.items():
            all_png1[exp] = dict()
            stats_table1 = pd.DataFrame(stats_table1)
            for i in tqdm(range(len(stats_table1))):
                value_list = stats_table1.iloc[i,1:-3].tolist()
                year_name = stats_table1.iloc[i,0]
                exp_name = exp
                insti_name = '集合'
                # 插值/掩膜/画图/保存
                mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, data_dir)
                
                # 转url
                png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
                all_png1[exp][year_name] = png_path
        
        # 历史-观测画图
        all_png2 = dict()
        stats_result_his = pd.DataFrame(stats_result_his)
        for i in tqdm(range(len(stats_result_his))):
            value_list = stats_result_his.iloc[i,1:-3].tolist()
            year_name = stats_result_his.iloc[i,0]
            exp_name = ''
            insti_name = ''
            # 插值/掩膜/画图/保存
            mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
            png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, data_dir)
            
            # 转url
            png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            all_png2[year_name] = png_path

    else: # 直接获取现成的，目前没做，所有图片路径都是None
        all_png = dict()
        all_png1 = dict()
        all_png2 = dict()

    result_dict['分布图']['预估单模式'] = all_png
    result_dict['分布图']['预估集合'] = all_png1
    result_dict['分布图']['历史'] = all_png2
    
    return result_dict


if __name__ == '__main__':
    data_json = dict()
    data_json['time_freq'] = 'Y'
    data_json['evaluate_times'] = '1970,1980' # 预估时段时间条
    data_json['refer_years'] = '2000,2024'# 参考时段时间条
    data_json['sta_ids'] = '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    data_json['cmip_type'] = 'original' # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ['Set']# 模式，列表：['CanESM5','CESM2']等
    data_json['plot'] = 0
    data_json['shp_path'] = r'C:/Users/MJY/Desktop/qhbh/zipdata/shp/qh/qh.shp'
    result = traffic_esti(data_json)
    