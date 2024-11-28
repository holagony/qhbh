# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:15:37 2024

@author: EDY
"""


import pandas as pd
from Module02.page_energy.wrapped.func06_read_model_data import read_model_data
import warnings
warnings.filterwarnings('ignore')


# 求得区域平均，区域指的是传进来的所有站点
def data_proce(df,processing_methods, additional_method=None):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['年份'] = df['Datetime'].dt.year
    
    if additional_method:
        processing_methods = {col: additional_method for col in processing_methods.keys()}
    else:
        processing_methods = processing_methods
        
    # 动态处理每个分组
    def process_group(group):
        processed = {
            'Station_Id_C': group['Station_Id_C'].iloc[0],
            'Datetime': group['Datetime'].iloc[0]
        }
        for element, method in processing_methods.items():
            if element in group.columns:
                processed[f'{element}'] = getattr(group[element], method)()
        return pd.Series(processed)
    
    grouped_data = df.groupby(['Station_Id_C', '年份']).apply(process_group).reset_index(drop=True)
    
    # 保存站点数据
    station_data=grouped_data.copy()
    station_data['年'] = station_data['Datetime'].dt.year
    station_data.drop(['Datetime'], axis=1, inplace=True) 
    
    return station_data


def model_factor_data_deal(data_dir, time_scale,insti_a,scene_a,station_id,var,ele,time_freq,time_freq_data,time_freq_main,stats_times,processing_methods):
    
    if time_freq== 'Y':
        # Y
        stats_times=stats_times
        
    elif time_freq in ['Q','M2']:
        stats_times=[stats_times,time_freq_data]
        
    elif time_freq== 'D2':
        stats_times=[stats_times,'0101,1231']


    refer_df=read_model_data(data_dir,time_scale,insti_a,scene_a,var,stats_times,time_freq,station_id)

    columns=refer_df.columns
    refer_df.reset_index(inplace=True)
    
    df = pd.DataFrame(columns=['Datetime', ele, 'Station_Id_C'])
    for i in columns:
        new_df=refer_df[['Datetime',i]]
        new_df.rename(columns={i:ele}, inplace=True)
        new_df['Station_Id_C']=i
        df = pd.concat([df, new_df], ignore_index=True)
    
    # 针对不同的要素进行不同的要素处理

    if time_freq_main != 'D':
        verify_station_data=data_proce(df,processing_methods)
    else:
        verify_station_data=df.copy()
        verify_data=df.copy()
        verify_data['Datetime'] = pd.to_datetime(verify_data['Datetime'])
        verify_data=verify_data.set_index(verify_data['Datetime'])
        verify_data.drop([ 'Station_Id_C','Datetime'], axis=1,inplace=True)
    
    
    return verify_station_data



if __name__ == '__main__':
    
    tas_paths = [
        r'D:\Project\qh\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19500101-19501231.nc',
        r'D:\Project\qh\original\daily\BCC-CSM2-MR\historical\tas\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19510101-19511231.nc',
        r'D:\Project\qh\original\daily\BCC-CSM2-MR\historical\tas\tas_day_BCC-CSM2-MR_historical_r3i1p1f1_gn_19520101-19521231.nc'
    ]
    
    # 插值站点信息
    sta_ids='51886,52737,52876'
    station_id = sta_ids.split(',')
    var='tas'
    ele='TEM_Avg'
    # 时间处理
    time_freq='Y'
    time_freq_data='0'    
    stats_times='1950,1952'
    
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
    
    verify_station_data,verify_data_deal=model_factor_data_deal(tas_paths,station_id,var,ele,time_freq,time_freq_data,stats_times,processing_methods)
























