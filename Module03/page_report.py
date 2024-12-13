# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:39:36 2024

@author: EDY
"""

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
from Module03.wrapped.func01_table_stats import table_stats_simple
from Module02.page_climate.wrapped.func02_table_stats_cmip import table_stats_simple_cmip
from Module03.wrapped.func02_plot import interp_and_mask, plot_and_save,line_chart_plot,bar_chart_plot,line_chart_cmip_plot
from Utils.config import cfg
from Utils.data_processing import data_processing
from Utils.data_loader_with_threads import get_database_data
from Utils.read_model_data import read_model_data

def find_non_min_keys(d):
    if not d:  
        return []
    min_key = min(d.keys())  
    return [key for key in d if key != min_key] 

def page_report(data_json):
    '''
    获取天擎数据，参数说明
    :param evaluate_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'
    
    :param refer_years: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'

    :param sta_ids: 传入的气象站点

    :param element：对应原型，传入的要素名称
        平均气温	TEM_Avg 
        最高气温	TEM_Max
        最低气温	TEM_Min
        降水量	PRE_Time_2020

    '''
    #%% 1.参数读取    
    refer_element = data_json['refer_element']  
    refer_years = data_json['refer_years']  
    evaluate_element = data_json['evaluate_element'] 
    evaluate_years = data_json['evaluate_years'] 
    cmip_type = data_json['cmip_type']  
    cmip_res = data_json.get('cmip_res') 
    cmip_model = data_json['cmip_model']  
    cmip_scenes = data_json['cmip_scenes']  
    sta_ids = data_json['sta_ids']  
    shp_path = data_json['shp_path']
    method = 'idw'

    if shp_path is not None:
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  
    
    # 2.参数处理
    degree = None
    uuid4 = uuid.uuid4().hex
    data_out = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_out):
        os.makedirs(data_out)
        os.chmod(data_out, 0o007 | 0o070 | 0o700)
    
    namelable=dict()
    namelable['TEM_Avg']='平均气温（℃）'
    namelable['TEM_Max']='平均最高气温（℃）'
    namelable['TEM_Min']='平均最低气温（℃）'
    namelable['PRE_Time_2020']='降雨量（mm）'

    
    var_dict = dict()
    var_dict['TEM_Avg'] = 'tas'
    var_dict['TEM_Max'] = 'tasmax'
    var_dict['TEM_Min'] = 'tasmin'
    var_dict['PRE_Time_2020'] = 'pr'
    

    # 直接读取excel
    res_d = dict()
    res_d['25'] = '0.25deg'
    res_d['50'] = '0.52deg'
    res_d['100'] = '1.00deg'
    
    if os.name == 'nt':
        data_dir = r'D:\Project\qh' # 本地
    else:
        if cmip_type == 'original':
            data_dir = '/model_data/station_data/csv' # 容器内
        elif cmip_type == 'delta':
            data_dir = '/model_data/station_data_delta/csv' # 容器内
            data_dir = os.path.join(data_dir, res_d[cmip_res])
        
    time_scale= 'yearly'
        
        
    # 结果字典
    result=dict()
    
    #%% 3. 气候特征-统计时段
    table_name = 'qh_climate_cmadaas_day'
    sta_ids_1 = tuple(sta_ids.split(','))
    tmp_dict=dict()
    station_name_flag=1
    station_name_dict=dict()
    result['统计']=dict()
    tmp_flag=0
    for element in refer_element:
        
        if element in ['TEM_Avg','TEM_Max','TEM_Min']:
            tmp_flag=1
            refer_df = get_database_data(sta_ids_1, element, table_name, 'Y', refer_years)
            
            df_unique = refer_df.drop_duplicates(subset='Station_Id_C')  # 删除重复行
            lon_list = df_unique['Lon'].tolist()
            lat_list = df_unique['Lat'].tolist()
            station_list_list = df_unique['Station_Name'].tolist()

            if station_name_flag==1:
                for i in range(len(df_unique)):
                    station_name_dict[df_unique['Station_Id_C'].iloc[i]]=df_unique['Station_Name'].iloc[i]
                station_name_flag=0 
                    
            refer_df['Station_Id_C'] =refer_df['Station_Name']
            
            refer_df = data_processing(refer_df, element, degree)
            stats_result_his = table_stats_simple(refer_df, element)
            
            
            # 变量要素
            result['统计'][element]=dict()
            
            result['统计'][element]['温度_均值']=stats_result_his['区域均值'].mean().round(1)
            result['统计'][element]['变率']=stats_result_his.iloc[-1,1:-3:].mean().round(1)
            result['统计'][element]['变率_最小']=stats_result_his.iloc[-1,1:-3:].min().round(1)
            result['统计'][element]['变率_最大']=stats_result_his.iloc[-1,1:-3:].max().round(1)
            result['统计'][element]['变率_站点_最大'] = stats_result_his.iloc[-1,1:-3:].to_frame().astype(float).max(skipna=True).iloc[0]
            result['统计'][element]['变率_站点_最大_站名'] = stats_result_his.iloc[-1,1:-3:].to_frame().astype(float).idxmax(skipna=True).iloc[0]
            result['统计'][element]['变率_站点_最小'] = stats_result_his.iloc[-1,1:-3:].to_frame().astype(float).min(skipna=True).iloc[0]
            result['统计'][element]['变率_站点_最小_站名'] = stats_result_his.iloc[-1,1:-3:].to_frame().astype(float).idxmin(skipna=True).iloc[0]
            # 对比
            tmp_dict[element]=result['统计'][element]['变率']
        
            # 折线图
            x= np.int32(stats_result_his['时间'].iloc[:-1:])
            y= stats_result_his['区域均值'].iloc[:-1:]
            data_out_png=os.path.join(data_out,element+'.png')
            line_chart_path=line_chart_plot(x,y,namelable[element],data_out_png)
            line_chart_path = line_chart_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            line_chart_path = line_chart_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['统计'][element]['折线图'] = line_chart_path
            
            #分布图
            value_list = stats_result_his[station_list_list].iloc[-1,::].tolist()
            mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
            png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, '统计', '统计',element , data_out,'气温变率℃/10年')
            png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['统计'][element]['分布图'] = png_path

        elif element in ['PRE_Time_2020']:
            
            refer_df = get_database_data(sta_ids_1, element, table_name, 'Y', refer_years)
            
            df_unique = refer_df.drop_duplicates(subset='Station_Id_C')  # 删除重复行
            lon_list = df_unique['Lon'].tolist()
            lat_list = df_unique['Lat'].tolist()
            station_list_list = df_unique['Station_Name'].tolist()

            if station_name_flag==1:
                for i in range(len(df_unique)):
                    station_name_dict[df_unique['Station_Id_C'].iloc[i]]=df_unique['Station_Name'].iloc[i]
                station_name_flag=0 
                
            refer_df['Station_Id_C'] =refer_df['Station_Name']
            refer_df = data_processing(refer_df, element, degree)
            stats_result_his = table_stats_simple(refer_df, element)
            
            
            # 变量要素
            result['统计'][element]=dict()
            result['统计'][element]['降雨量_均值']=stats_result_his['区域均值'].mean().round(1)
            result['统计'][element]['变率']=stats_result_his.iloc[-1,1:-3:].mean().round(1)
            result['统计'][element]['降雨量_最大'] = np.round(stats_result_his['区域均值'].max(),1)
            result['统计'][element]['降雨量_最大_年'] = stats_result_his['时间'].loc[stats_result_his['区域均值'].idxmax()]
            result['统计'][element]['降雨量_最小'] = np.round(stats_result_his['区域均值'].min(),1)
            result['统计'][element]['降雨量_最小_年'] = stats_result_his['时间'].loc[stats_result_his['区域均值'].idxmin()]
                        
            # 柱状图
            x= np.int32(stats_result_his['时间'].iloc[:-1:])
            y= stats_result_his['区域均值'].iloc[:-1:]
            data_out_png=os.path.join(data_out,element+'.png')
            bar_chart_path=bar_chart_plot(x,y,namelable[element],data_out_png)
            bar_chart_path = bar_chart_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            bar_chart_path = bar_chart_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['统计'][element]['折线图'] = bar_chart_path
            
            #分布图
            value_list = stats_result_his[station_list_list].iloc[-1,::].tolist()
            mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
            png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, '统计', '统计',element , data_out,'降水量变率mm/10年')
            png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['统计'][element]['分布图'] = png_path

    if tmp_flag==1:
        key_mapping = {'TEM_Avg': '平均温度','TEM_Max': '最高温度','TEM_Min': '最低温度'}
        non_min_keys = find_non_min_keys(tmp_dict)
        chinese_keys = [key_mapping.get(key, key) for key in non_min_keys]
        result['统计']['TEM']=chinese_keys
            
            
    #%% 4. 气候变化趋势预估
    result['评估']=dict()
    sta_ids_2 =sta_ids.split(',')
    for element in evaluate_element:
        
        if element in ['TEM_Avg','TEM_Max','TEM_Min']:
            
            refer_df = get_database_data(sta_ids_1, element, table_name, 'Y', '1981,2014')
            df_unique = refer_df.drop_duplicates(subset='Station_Id_C')  # 删除重复行
            lon_list = df_unique['Lon'].tolist()
            lat_list = df_unique['Lat'].tolist()
            refer_df['Station_Id_C'] =refer_df['Station_Name']
            refer_df = data_processing(refer_df, element, degree)
            stats_result_his_cmip = table_stats_simple(refer_df, element)
             
            tmp_dict=dict()
            tmp_maxin=dict()
            cmip_data=pd.DataFrame()
            
            for exp in cmip_scenes:
                for insti in cmip_model:
                    excel_data = read_model_data(data_dir,time_scale,insti,exp,var_dict[element],evaluate_years,'Y',sta_ids_2)
                    for columns1 in excel_data.columns:
                        excel_data.rename(columns={columns1:station_name_dict.get(columns1, columns1)}, inplace=True)
                    excel_data_1=excel_data.copy()
                    stats_result_cmip = table_stats_simple(excel_data, element,'cmip')
                    
                    # 变量要素
                    result['评估'][element]=dict()
                    result['评估'][element]['变率']=stats_result_cmip.iloc[-1,1:-3:].mean().round(1)
                    
                    # 对比
                    cmip_data[exp]=stats_result_cmip['区域均值'].iloc[:-1:]-stats_result_his_cmip['区域均值'].mean().round(1)
                    tmp_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmax(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).max(skipna=True).iloc[0]
                    tmp_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmin(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).min(skipna=True).iloc[0]
                    tmp_maxin[exp+'变率_最小']=np.round(stats_result_cmip.iloc[-1,1:-3:].min(),1)
                    tmp_maxin[exp+'变率_最大']=np.round(stats_result_cmip.iloc[-1,1:-3:].max(),1)
                    
                    #分布图
                    stats_result_cmip_jp = table_stats_simple(excel_data_1-stats_result_his_cmip['区域均值'].mean().round(1), element,'cmip')
                    value_list = stats_result_cmip_jp[station_list_list].iloc[-1,::].tolist()
                    mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                    png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp, insti,element , data_out,'年平均气温距平变化率（℃/10年）')
                    png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                    png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
                    result['评估'][element]['分布图'] = png_path
                    
            result['评估'][element]['变率_最大']=max(tmp_maxin.values())
            result['评估'][element]['变率_最小']=min(tmp_maxin.values())
            result['评估'][element]['变率_站点_最大'] = max(tmp_dict.values())
            result['评估'][element]['变率_站点_最大_站名'] = max(tmp_dict.keys())
            result['评估'][element]['变率_站点_最小'] = min(tmp_dict.values())
            result['评估'][element]['变率_站点_最小_站名'] = max(tmp_dict.keys())
                    
                    
            # 折线图
            x= np.int32(stats_result_cmip['时间'].iloc[:-1:])
            data_out_png=os.path.join(data_out,element+'.png')
            line_chart_path=line_chart_cmip_plot(x,cmip_data,'气温距平（℃）',data_out_png)
            line_chart_path = line_chart_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            line_chart_path = line_chart_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['评估'][element]['折线图'] = line_chart_path
                    
        elif element in ['PRE_Time_2020']:
            
            refer_df = get_database_data(sta_ids_1, element, table_name, 'Y', '1981,2014')
            df_unique = refer_df.drop_duplicates(subset='Station_Id_C')  # 删除重复行
            lon_list = df_unique['Lon'].tolist()
            lat_list = df_unique['Lat'].tolist()
            refer_df['Station_Id_C'] =refer_df['Station_Name']
            refer_df = data_processing(refer_df, element, degree)
            stats_result_his_cmip = table_stats_simple(refer_df, element)
             

            pre_dict=dict()
            pre_maxin=dict()
            for exp in cmip_scenes:
                for insti in cmip_model:
                    excel_data = read_model_data(data_dir,time_scale,insti,exp,var_dict[element],evaluate_years,'Y',sta_ids_2)
                    for columns1 in excel_data.columns:
                        excel_data.rename(columns={columns1:station_name_dict.get(columns1, columns1)}, inplace=True)
                    stats_result_cmip = table_stats_simple(excel_data, element,'cmip')
                    
                    # 变量要素
                    result['评估'][element]=dict()
                    result['评估'][element]['变率']=stats_result_cmip.iloc[-1,1:-3:].mean().round(1)
                    
                    # 对比
                    pre_mean=stats_result_his_cmip['区域均值'].mean().round(1)
                    cmip_data[exp]=np.round(((stats_result_cmip['区域均值'].iloc[:-1:]-pre_mean)/pre_mean)*100,2)
                    pre_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmax(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).max(skipna=True).iloc[0]
                    pre_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmin(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).min(skipna=True).iloc[0]
                    pre_maxin[exp+'变率_最小']=np.round(stats_result_cmip.iloc[-1,1:-3:].min(),1)
                    pre_maxin[exp+'变率_最大']=np.round(stats_result_cmip.iloc[-1,1:-3:].max(),1)
                    
                    #分布图
                    stats_result_cmip_jp = table_stats_simple(((excel_data_1-pre_mean)/pre_mean)*100, element,'cmip')
                    value_list = stats_result_cmip_jp[station_list_list].iloc[-1,::].tolist()
                    mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                    png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp, insti,element , data_out,'年降水距平百分率变化率（%/10年）')
                    png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                    png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
                    result['评估'][element]['分布图'] = png_path
            
            result['评估'][element]['变率_最大']=max(pre_maxin.values())
            result['评估'][element]['变率_最小']=min(pre_maxin.values())
            result['评估'][element]['变率_站点_最大'] = max(pre_dict.values())
            result['评估'][element]['变率_站点_最大_站名'] = max(pre_dict.keys())
            result['评估'][element]['变率_站点_最小'] = min(pre_dict.values())
            result['评估'][element]['变率_站点_最小_站名'] = max(pre_dict.keys())
            
            # 折线图
            x= np.int32(stats_result_cmip['时间'].iloc[:-1:])
            data_out_png=os.path.join(data_out,element+'.png')
            line_chart_path=line_chart_cmip_plot(x,cmip_data,'降水距平百分率（%）',data_out_png)
            line_chart_path = line_chart_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            line_chart_path = line_chart_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['评估'][element]['折线图'] = line_chart_path

    return result


if __name__ == '__main__':
    data_json = dict()
    data_json['refer_element'] = ['TEM_Avg','TEM_Max']# 可多选
    data_json['refer_years'] = '1995,2014'  # 参考时段时间条

    data_json['evaluate_element'] = ['TEM_Avg','TEM_Max']# 可多选
    data_json['evaluate_years'] = "2010,2025"  # 预估时段时间条
    
    data_json['cmip_type'] = 'original'  # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None  # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ['Set']  # 模式，列表：['CanESM5','CESM2']/
    data_json['cmip_scenes'] = ['ssp126','ssp245']  # 模式，列表：['CanESM5','CESM2']/
    data_json['sta_ids'] = '51886,52602,52633,52645,52657,52707,52713'
    data_json['shp_path'] = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\州界.shp'
    res_table = climate_esti(data_json)
    
