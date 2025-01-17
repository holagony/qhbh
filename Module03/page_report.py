# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:39:36 2024

@author: EDY
"""

import os
import uuid
import numpy as np
import pandas as pd
from Module03.wrapped.func01_table_stats import table_stats_simple
from Module03.wrapped.func02_plot import interp_and_mask, plot_and_save,line_chart_plot,bar_chart_plot,line_chart_cmip_plot
from Utils.config import cfg
from Utils.data_processing import data_processing
from Utils.data_loader_with_threads import get_database_data
from Module02.page_energy.wrapped.func06_read_model_data import read_model_data

def find_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None  # 如果没有找到对应的键，返回None

def get_keys_except_one(d, key_to_exclude):
    return [key for key in d.keys() if key != key_to_exclude]

def station_name_deal(s):
    if '国' in s:
        s=s[:s.find('国')]
    else:
        s=s
    return s

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
    area=data_json.get('area','该')
    refer_element = data_json['refer_element']  
    refer_years = data_json['refer_years']  
    evaluate_element = data_json['evaluate_element'] 
    evaluate_years = data_json['evaluate_years'] 
    cmip_type = data_json.get('cmip_type','original')  
    cmip_res = data_json.get('cmip_res','25') 
    cmip_model = data_json['cmip_model']  
    cmip_scenes = data_json['cmip_scenes']  
    sta_ids = data_json['sta_ids']  
    shp_path = data_json['shp_path']
    method = 'idw'

    #print(refer_years)   
    #print(evaluate_years)
    #print(shp_path)
    #print(sta_ids)

    if isinstance(cmip_model, str):
        cmip_model = cmip_model.split(',')
        
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

    namelable2=dict()
    namelable2['TEM_Avg']='平均气温'
    namelable2['TEM_Max']='最高气温'
    namelable2['TEM_Min']='最低气温'
        
    var_dict = dict()
    var_dict['TEM_Avg'] = 'tas'
    var_dict['TEM_Max'] = 'tasmax'
    var_dict['TEM_Min'] = 'tasmin'
    var_dict['PRE_Time_2020'] = 'pr'
    

    # 直接读取excel
    res_d = dict()
    res_d['25'] = '0.25deg'
    res_d['50'] = '0.50deg'
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
    result_dict=dict()

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
            result['统计'][element]['变率_站点_最大_站名'] = station_name_deal(stats_result_his.iloc[-1,1:-3:].to_frame().astype(float).idxmax(skipna=True).iloc[0])
            result['统计'][element]['变率_站点_最小'] = stats_result_his.iloc[-1,1:-3:].to_frame().astype(float).min(skipna=True).iloc[0]
            result['统计'][element]['变率_站点_最小_站名'] = station_name_deal(stats_result_his.iloc[-1,1:-3:].to_frame().astype(float).idxmin(skipna=True).iloc[0])
            # 对比
            tmp_dict[element]=result['统计'][element]['变率']
        
            # 折线图
            x= np.int32(stats_result_his['时间'].iloc[:-1:])
            y= stats_result_his['区域均值'].iloc[:-1:]
            data_out_png=os.path.join(data_out,'统计'+element+'.png')
            line_chart_path=line_chart_plot(x,y,namelable[element],data_out_png)
            line_chart_path = line_chart_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            line_chart_path = line_chart_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['统计'][element]['折线图'] = line_chart_path
            
            #分布图
            value_list = stats_result_his[station_list_list].iloc[-1,::].tolist()
            mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
            png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, '统计', '统计',element , data_out,f'{namelable2[element]}/10年')
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
            data_out_png=os.path.join(data_out,'统计'+element+'.png')
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
        non_min_keys = get_keys_except_one(tmp_dict,find_key_by_value(tmp_dict, min(tmp_dict.values())))
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
            result['评估'][element]=dict()

            for exp in cmip_scenes:
                for insti in cmip_model:
                    excel_data = read_model_data(data_dir,time_scale,insti,exp,var_dict[element],evaluate_years,'Y',sta_ids_2)
                    for columns1 in excel_data.columns:
                        excel_data.rename(columns={columns1:station_name_dict.get(columns1, columns1)}, inplace=True)
                    excel_data_1=excel_data.copy()
                    stats_result_cmip = table_stats_simple(excel_data, element,'cmip')
                    
                    # 变量要素
                    result['评估'][element][exp+'变率']=stats_result_cmip.iloc[-1,1:-3:].mean().round(1)
                    
                    # 对比
                    cmip_data[exp]=stats_result_cmip['区域均值'].iloc[:-1:]-stats_result_his_cmip['区域均值'].mean().round(1)
                    tmp_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmax(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).max(skipna=True).iloc[0]
                    tmp_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmin(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).min(skipna=True).iloc[0]
                    tmp_maxin[exp+'变率_最小']=np.round(stats_result_cmip.iloc[-1,1:-3:].min(),1)
                    tmp_maxin[exp+'变率_最大']=np.round(stats_result_cmip.iloc[-1,1:-3:].max(),1)
                    
                    #分布图
                    stats_result_cmip_jp = table_stats_simple(excel_data_1-stats_result_his_cmip['区域均值'].mean().round(1), element,'cmip')
                    # value_list = stats_result_cmip_jp[station_list_list].iloc[-1,::].tolist()
                    
                    value_list = []
                    for station in station_list_list:
                        if station in stats_result_cmip_jp.columns:
                            value_list.append(stats_result_cmip_jp[station].iloc[-1])
                            # print(station)
                        else:
                            value_list.append(np.nan)
                            
                    mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                    png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp, insti,element , data_out,f'年{namelable2[element]}距平变化率（℃/10年）')
                    png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                    png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
                    result['评估'][element][exp+'分布图'] = png_path
                    
            result['评估'][element]['变率_最大']=max(tmp_maxin.values())
            result['评估'][element]['变率_最小']=min(tmp_maxin.values())
            result['评估'][element]['变率_站点_最大'] = max(tmp_dict.values())
            result['评估'][element]['变率_站点_最大_站名'] = station_name_deal(find_key_by_value(tmp_dict,max(tmp_dict.values())))
            result['评估'][element]['变率_站点_最小'] = min(tmp_dict.values())
            result['评估'][element]['变率_站点_最小_站名'] = station_name_deal(find_key_by_value(tmp_dict,min(tmp_dict.values())))
                    
                    
            # 折线图
            x= np.int32(stats_result_cmip['时间'].iloc[:-1:])
            data_out_png=os.path.join(data_out,'评估'+element+'.png')
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
            result['评估'][element]=dict()

            for exp in cmip_scenes:
                for insti in cmip_model:
                    # break
                    excel_data = read_model_data(data_dir,time_scale,insti,exp,var_dict[element],evaluate_years,'Y',sta_ids_2)
                    for columns1 in excel_data.columns:
                        excel_data.rename(columns={columns1:station_name_dict.get(columns1, columns1)}, inplace=True)
                    stats_result_cmip = table_stats_simple(excel_data, element,'cmip')
                    
                    # 变量要素
                    result['评估'][element][exp+'变率']=stats_result_cmip.iloc[-1,1:-3:].mean().round(1)
                    
                    # 对比
                    pre_mean=stats_result_his_cmip['区域均值'].mean().round(1)
                    cmip_data[exp]=np.round(((stats_result_cmip['区域均值'].iloc[:-1:]-pre_mean)/pre_mean)*100,2)
                    pre_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmax(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).max(skipna=True).iloc[0]
                    pre_dict[stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).idxmin(skipna=True).iloc[0]]=stats_result_cmip.iloc[-1,1:-3:].to_frame().astype(float).min(skipna=True).iloc[0]
                    pre_maxin[exp+'变率_最小']=np.round(stats_result_cmip.iloc[-1,1:-3:].min(),1)
                    pre_maxin[exp+'变率_最大']=np.round(stats_result_cmip.iloc[-1,1:-3:].max(),1)
                    
                    #分布图
                    stats_result_cmip_jp = table_stats_simple(((excel_data_1-pre_mean)/pre_mean)*100, element,'cmip')
                    # value_list = stats_result_cmip_jp[station_list_list].iloc[-1,::].tolist()
                    
                    value_list = []
                    for station in station_list_list:
                        if station in stats_result_cmip_jp.columns:
                            value_list.append(stats_result_cmip_jp[station].iloc[-1])
                            # print(station)
                        else:
                            value_list.append(np.nan)
                            
                    mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                    png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp, insti,element , data_out,'年降水距平百分率变化率（%/10年）')
                    png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                    png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
                    result['评估'][element][exp+'分布图'] = png_path
            
            result['评估'][element]['变率_最大']=max(pre_maxin.values())
            result['评估'][element]['变率_最小']=min(pre_maxin.values())
            result['评估'][element]['变率_站点_最大'] = max(pre_dict.values())
            result['评估'][element]['变率_站点_最大_站名'] =  station_name_deal(find_key_by_value(pre_dict,max(pre_dict.values())))
            result['评估'][element]['变率_站点_最小'] = min(pre_dict.values())
            result['评估'][element]['变率_站点_最小_站名'] =  station_name_deal(find_key_by_value(pre_dict,min(pre_dict.values())))
            
            # 折线图
            x= np.int32(stats_result_cmip['时间'].iloc[:-1:])
            data_out_png=os.path.join(data_out,'评估'+element+'.png')
            line_chart_path=line_chart_cmip_plot(x,cmip_data,'降水距平百分率（%）',data_out_png)
            line_chart_path = line_chart_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
            line_chart_path = line_chart_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
            result['评估'][element]['折线图'] = line_chart_path
            
            
            
    # 文字生成
    refer_start_year=refer_years.split(',')[0]
    refer_end_year=refer_years.split(',')[1]

    
    result_dict['统计']=dict()
    # 统计温度
    temp_dict=dict()
    temp_dict['TEM_Avg']='平均气温'
    temp_dict['TEM_Max']='平均最高气温'
    temp_dict['TEM_Min']='平均最低气温'

    result_dict['统计']['温度']=dict()
    result_dict['统计']['温度']['图片']=dict()
    result_dict['统计']['温度']['图片']['分布图']=dict()
    result_dict['统计']['温度']['图片']['折线图']=dict()

    text_1=''
    text_2=''
    text_3=''
    for i, element in enumerate(refer_element):
        if element in ['TEM_Avg','TEM_Max','TEM_Min']:
            text_1=text_1+'年'+temp_dict[element]+'为'+str(result['统计'][element]['温度_均值'])+'℃,'
            text_2=text_2+str(result['统计'][element]['变率'])+'℃/10年,'
            text_3=text_3+f"{area}各地年{temp_dict[element]}变率在{str(result['统计'][element]['变率_最小'])}～{str(result['统计'][element]['变率_最大'])}℃/10年之间，其中{str(result['统计'][element]['变率_站点_最大_站名'])}升温率最高为{str(result['统计'][element]['变率_站点_最大'])}℃/10年以上，而{str(result['统计'][element]['变率_站点_最小_站名'])}升温率最低为{str(result['统计'][element]['变率_站点_最小'])}℃/10年。"

                
            result_dict['统计']['温度']['图片']['分布图'][element]=result['统计'][element]['分布图']
            result_dict['统计']['温度']['图片']['折线图'][element]=result['统计'][element]['折线图']
        
    result_dict['统计']['温度']['文字']=f"{refer_start_year}—{refer_end_year}年，{area}{text_1}升温率分别为{text_2}以{','.join(result['统计']['TEM'])}的上升趋势最为明显。{text_3}"

    
    # 统计降雨
    result_dict['统计']['降雨']=dict()
    result_dict['统计']['降雨']['图片']=dict()
    result_dict['统计']['降雨']['图片']['分布图']=dict()
    result_dict['统计']['降雨']['图片']['折线图']=dict()
    
    if result['统计']['PRE_Time_2020']['变率']>0:
        ptr_q='增加'
    else:
        ptr_q='减少'
        result['统计']['PRE_Time_2020']['变率']=np.abs(result['统计']['PRE_Time_2020']['变率'])
        
    result_dict['统计']['降雨']['文字']=f"{refer_start_year}—{refer_end_year}年，{area}年降水量为{str(result['统计']['PRE_Time_2020']['降雨量_均值'])}毫米，呈显著{ptr_q}趋势，平均每10年{ptr_q}{str(result['统计']['PRE_Time_2020']['变率'])}毫米。{str(result['统计']['PRE_Time_2020']['降雨量_最小_年'])}年降水量历史最小值为{str(result['统计']['PRE_Time_2020']['降雨量_最小'])}mm。{str(result['统计']['PRE_Time_2020']['降雨量_最大_年'])}年降水量达到历史最大值为{str(result['统计']['PRE_Time_2020']['降雨量_最大'])}mm。"
    
    result_dict['统计']['降雨']['图片']['分布图']['PRE_Time_2020']=result['统计']['PRE_Time_2020']['分布图']
    result_dict['统计']['降雨']['图片']['折线图']['PRE_Time_2020']=result['统计']['PRE_Time_2020']['折线图']



    result_dict['评估']=dict()
    
    # 评估温度
    result_dict['评估']['温度']=dict()
    result_dict['评估']['温度']['图片']=dict()
    result_dict['评估']['温度']['图片']['折线图']=dict()
    result_dict['评估']['温度']['图片']['分布图']=dict()
    
    evaluate_start_year=evaluate_years.split(',')[0]
    evaluate_end_year=evaluate_years.split(',')[1]
    
    model_dict=dict()
    model_dict['Set']='CIMP6多个全球气候模式'
    model_dict['RCM_BCC']='RCM_BCC模式'
    model_dict['MPI-ESM1-2-LR']='MPI-ESM1-2-LR模式'
    model_dict['NESM3']='NESM3模式'
    
    text_1=''
    for i, element in enumerate(evaluate_element):
        if element in ['TEM_Avg','TEM_Max','TEM_Min']:
            text_2=''
            for exp in cmip_scenes:
                text_2=text_2+str(result['评估'][element][exp+'变率'])+'℃/10年,'
  
                result_dict['评估']['温度']['图片']['分布图'][element+exp]=result['评估'][element][exp+'分布图']
            text_1=text_1+f"{evaluate_start_year}—{evaluate_end_year}年{area}年，气候倾向率分别为为{text_2}各地年{temp_dict[element]}变率在{str(result['评估'][element]['变率_最小'])}～{str(result['评估'][element]['变率_最大'])}℃/10 年之间，其中{str(result['评估'][element]['变率_站点_最大_站名'])}升温率最高为{str(result['评估'][element]['变率_站点_最大'])}℃/10年以上，而{str(result['评估'][element]['变率_站点_最小_站名'])}升温率最低为{str(result['评估'][element]['变率_站点_最小'])}℃/10年。"
            
            result_dict['评估']['温度']['图片']['折线图'][element]= result['评估'][element]['折线图']

    result_dict['评估']['温度']['文字']=f"{model_dict[cmip_model[0]]}对未来不同情景下（{'、'.join(cmip_scenes)}）{area}气温变化的预估结果表明：在不同排放情景下{text_1}"

    # 评估降雨
    result_dict['评估']['降雨']=dict()
    result_dict['评估']['降雨']['图片']=dict()
    result_dict['评估']['降雨']['图片']['折线图']=dict()
    result_dict['评估']['降雨']['图片']['分布图']=dict()

    text_2=''
    for exp in cmip_scenes:
        text_2=text_2+str(result['评估']['PRE_Time_2020'][exp+'变率'])+'mm/10年,'
        result_dict['评估']['降雨']['图片']['分布图'][element+exp]=result['评估']['PRE_Time_2020'][exp+'分布图']
        
    
    result_dict['评估']['降雨']['图片']['折线图'][element]= result['评估']['PRE_Time_2020']['折线图']
    result_dict['评估']['降雨']['文字']=f"{model_dict[cmip_model[0]]}对未来不同情景下（{'、'.join(cmip_scenes)}）{area}降水量变化的预估结果表明：在不同排放情景下，{evaluate_start_year}-{evaluate_end_year}年气候倾向率分别为{text_2}其中{str(result['评估']['PRE_Time_2020']['变率_站点_最大_站名'])}最高为{str(result['评估']['PRE_Time_2020']['变率_站点_最大'])}mm/10年以上，而{str(result['评估']['PRE_Time_2020']['变率_站点_最小_站名'])}最低为{str(result['评估']['PRE_Time_2020']['变率_站点_最小'])}mm/10年。"
    

    return result_dict


if __name__ == '__main__':
    data_json = dict()
    data_json['area'] = '青海湖流域'# 可多选
    data_json['refer_element'] = ["TEM_Avg","TEM_Max","TEM_Min","PRE_Time_2020"]# 可多选
    data_json['refer_years'] = "2014,2024"  # 参考时段时间条

    data_json['evaluate_element'] = ["TEM_Avg","TEM_Max","TEM_Min","PRE_Time_2020"]# 可多选
    data_json['evaluate_years'] = "2014,2024"  # 预估时段时间条
    
    data_json['cmip_type'] = 'original'  # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None  # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ["Set"]  # 模式，列表：['CanESM5','CESM2']/
    data_json['cmip_scenes'] = ["ssp126", "ssp245", "ssp585"]  # 模式，列表：['CanESM5','CESM2']/
    data_json['sta_ids'] = '52745,52754,52853,52856'
    data_json['shp_path'] = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\03-泛共和盆地\泛共和盆地.shp'
    
    result_dict= page_report(data_json)