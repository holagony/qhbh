# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:23:59 2024

@author: EDY
"""

import pandas as pd
import numpy as np
from Module02.page_energy.wrapped.func00_function import choose_mod_path
from Module02.page_ice.wrapped.func01_factor_data_deal import factor_data_deal
from Module02.page_ice.wrapped.func02_ice_evaluate_data_deal import ice_evaluate_data_deal
from Module02.page_ice.wrapped.func03_model_factor_data_deal import model_factor_data_deal

def ice_table_def(data_json):

    main_element=data_json['main_element']
    sta_ids=data_json['sta_ids']
    time_freq_main=data_json['time_freq_main']
    time_freq_main_data=data_json['time_freq_main_data']
    refer_times=data_json['refer_times']
    stats_times=data_json['stats_times']
    data_cource=data_json['data_cource']
    insti=data_json['insti']
    scen=data_json['scen']
    res=data_json['res']
    time_scale=data_json['time_scale']
    factor_element=data_json['factor_element']
    factor_time_freq=data_json['factor_time_freq']
    factor_time_freq_data=data_json['factor_time_freq_data']
    verify_time=data_json['verify_time']

    data_dir=r'D:\Project\qh\Evaluate_Energy\data'
    
    
    #%% 固定字典表
    # 分辨率
    res_d=dict()
    res_d['10']='0.10deg'
    res_d['25']='0.25deg'
    res_d['50']='0.50deg'
    res_d['100']='1deg'

    # 模型站点要素名对照表
    model_ele_dict=dict()
    model_ele_dict['TEM_Avg']='tas'
    model_ele_dict['PRE_Time_2020']='pr'

    # 不同的要素不同的处理方法,针对日尺度之外的处理
    resample_max = ['TEM_Max', 'PRS_Max', 'WIN_S_Max', 'WIN_S_Inst_Max', 'GST_Max', 'huangku']

    resample_min = ['TEM_Min', 'PRS_Min', 'GST_Min', 'RHU_Min', 'fanqing']

    resample_sum = ['SSH','PRE_Time_2020', 'PRE_Days', 'EVP_Big', 'EVP', 'EVP_Taka', 'PMET','sa','rainstorm','light_snow','snow',
                    'medium_snow','heavy_snow','severe_snow','Hail_Days','Hail','GaWIN',
                    'GaWIN_Days','SaSt','SaSt_Days','FlSa','FlSa_Days','FlDu','FlDu_Days',
                    'Thund','Thund_Days''high_tem','drought','light_drought','medium_drought',
                    'heavy_drought','severe_drought','Accum_Tem']

    resample_mean = ['TEM_Avg', 'PRS_Avg', 'WIN_S_2mi_Avg', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Avg_5cm', 'GST_Avg_10cm', 
                     'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 
                     'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 'SSH', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg', 'Cov', 'pmet']

    processing_methods = {element: 'mean' for element in resample_mean}
    processing_methods.update({element: 'sum' for element in resample_sum})
    processing_methods.update({element: 'max' for element in resample_max})
    processing_methods.update({element: 'min' for element in resample_min})
    
    
    #%% 验证期 因子数据 评估数据
    verify_station,verify_data=factor_data_deal(factor_element,verify_time,sta_ids,factor_time_freq,factor_time_freq_data,time_freq_main,processing_methods)
    verify_data=verify_data.reset_index()
    verify_data = verify_data.rename(columns={'年份': 'Datetime'})
    verify_data = verify_data.set_index(verify_data['Datetime'].astype(str))
    
    verify_evaluate,verify_evaluate_station=ice_evaluate_data_deal(main_element,verify_time,sta_ids,time_freq_main,time_freq_main_data)
    evaluate=pd.merge(verify_data,verify_evaluate, left_index=True, right_index=True, how='inner')
    evaluate.dropna(inplace=True)
    
    
    #%% 表格数据
    
    # 观测
    result_0=verify_evaluate_station.reset_index()
    result_0.columns.values[0] = '年'
    result_0=result_0[result_0['年'].astype(float).isin(evaluate['Datetime'])]
    
    # 模拟观测
    station_id_c=verify_station['Station_Id_C'].unique()
    result_1=pd.DataFrame(index=np.arange(len(evaluate)),columns=station_id_c)
    for i in station_id_c:
        station_i=verify_station[verify_station['Station_Id_C']==i]
        station_i=station_i[independent_columns]
        result_1[i]=model.predict(station_i)
        
    datetime_column = evaluate.reset_index(drop=True)['Datetime']
    result_1.insert(0, '年', datetime_column)
    result_1[result_1<0]=0
    
    verify_data_i=verify_data[independent_columns]
    result_1['区域平均']=model.predict(verify_data_i)
    
    
if __name__ == '__main__':
    
    
    data_json = dict()
    data_json['main_element']='FRS_DEPTH'  # 评估要素
    data_json['sta_ids']='51886,52737,52876' # 站点信息
    data_json['time_freq_main']='Y' # 评估要素时间尺度
    data_json['time_freq_main_data']='0'
    data_json['refer_times'] = '2020,2022' # 参考时段
    data_json['stats_times'] = '2020,2040' # 评估时段
    data_json['data_cource'] = 'original' # 模式信息
    data_json['insti']= 'BCC-CSM2-MR,CanESM5'
    data_json['scen']=['ssp126','ssp245']
    data_json['res'] ='1'
    data_json['time_scale']='daily'
    data_json['data_dir']=r'D:\Project\qh\Evaluate_Energy\data'
    data_json['factor_element']='TEM_Avg,PRE_Time_2020'     # 关键因子
    data_json['factor_time_freq']='Y,Q' # 关键因子时间尺度
    data_json['factor_time_freq_data']=['0','3,4,5']
    data_json['verify_time']='2020,2021' # 验证日期

    # 要素变量
    data_json['参数']=dict()
    data_json['参数']['截距']=1
    data_json['参数']['TEM_Avg']=2
    data_json['参数']['PRE_Time_2020']=3
    
    