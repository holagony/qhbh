# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:49:11 2025

@author: EDY
"""

import pandas as pd
import numpy as np
from Module02.page_energy.wrapped.func06_read_model_data import read_model_data
from Utils.config import cfg

def calculate_expression(tas_value, t_opt):
    if -13 <= tas_value - t_opt < 10:
        return (1.1814 / (1 + np.exp(0.2 * (t_opt - 10 - tas_value)))) * (1 / (1 + np.exp(0.3 * (tas_value - 10 - t_opt))))
    else:
        return (1.1814 / (1 + np.exp(0.2 * (-10)))) * (1 / (1 + np.exp(0.3 * (-10))))/2

# 潜在蒸散量
def calculate_ep0(tas_value):
    H=(tas_value/5)**1.514
    A=6.75*10e-7*H**3-7.71*10e-5*H**2+1.792*10e-2*H+0.49
    if tas_value <=0:
        PET=0
    else:
        PET=16*(10*tas_value/H)**A
    return PET



def cesva_pre(element,data_dir,time_scale,insti,scene,stats_times,time_freq,station_ids,station_dict):
    
    # 读取数据
    cesva_file=pd.read_csv(cfg.FILES.CESVA)
    df_rsds=read_model_data(data_dir,time_scale,insti,scene,'rsds',stats_times,time_freq,station_ids)
    
    if time_scale== 'monthly':
        df_rsds = df_rsds*24 * 3600*31 / 1e6
    elif time_scale== 'yearly':
        df_rsds = df_rsds*24 * 3600*360 / 1e6
        

    df_tas=read_model_data(data_dir,time_scale,insti,scene,'tas',stats_times,time_freq,station_ids)
    df_pr=read_model_data(data_dir,time_scale,insti,scene,'pr',stats_times,time_freq,station_ids)

    # 植被吸收光合有效辐射（APAR）的计算
    df_apar=pd.DataFrame(index=df_rsds.index,columns=df_rsds.columns)
    for column in  df_rsds.columns:
        
        fpar=cesva_file[cesva_file['station id'].astype(int)==int(column)]['fpar2'].iloc[0]
        df_apar[column]=df_rsds[column]*fpar*0.5
         
    # 温度胁迫因子
    df_t1=pd.DataFrame(index=df_tas.index,columns=df_tas.columns)
    df_t2=df_t1.copy()

    for column in  df_tas.columns:
        
        t_opt=cesva_file[cesva_file['station id'].astype(int)==int(column)]['tem'].iloc[0]
        df_t1[column]=0.8+0.02*t_opt-0.0005*t_opt**2
        df_t1.loc[df_tas[column] <= -10, column] = 0

        df_t2[column] = df_tas[column].apply(lambda x: calculate_expression(x, t_opt))

    df_t=df_t1*df_t2
    
    # 水分胁迫因子
    df_w=pd.DataFrame(index=df_tas.index,columns=df_tas.columns)
    for column in  df_rsds.columns:
        
        df_w1=pd.DataFrame()
        # 高桥实际蒸发？
        df_w1['AET']= 3100*df_tas[column].astype(float)/(3100+1.8*(df_pr[column].astype(float)**2)*np.exp((-34.4*df_tas[column].astype(float))/(235+df_tas[column].astype(float))))
       
        # 潜在潜在蒸散量
        df_w1['EP_O']=df_tas[column].apply(lambda x: calculate_ep0(x))
        df_w1['PET']=(df_w1['AET']+df_w1['EP_O'])/2
        df_w1['df_w'] = df_w1.apply(lambda row: (row['AET'] / row['PET']) if row['PET'] > 0 else 0, axis=1)
        df_w1['df_w'] = df_w1['df_w'].clip(0, 1)
        
        df_w[column]= 0.5+0.5*df_w1['df_w']
        
    result=df_apar*df_t*df_w*0.542

    # 转为年
    result=result.resample('Y').sum().round(1)

    result.index = result.index.strftime('%Y')
    result.reset_index(inplace=True)
    result.columns.values[0] = '年'

    return result
    
    
if __name__=='__main__':
    
    data_dir=r'D:\Project\qh'
    time_scale= 'monthly'
    insti='NESM3'
    scene='ssp126'
    var='tas'
    station_ids=['51886','52602','52633','52645','52657','52707','52713']
    
    station_ids='51886,52602,52633,52645,52657,52707,52713,52737,52745,52754,52765,52818,52825,52833,52836,52842,52853,52855,52856,52862,52863,52866,52868,52869,52874,52875,52876,52877,52908,52943,52955,52957,52963,52968,52972,52974,56004,56016,56018,56021,56029,56033,56034,56043,56045,56046,56065,56067,56125,56151'
    station_ids=station_ids.split(',')
    
    stats_times='2030,2040'
    time_freq= 'Y'
    
    stats_times=['2011,2040','12,1,2']
    time_freq= 'Q'
    
    stats_times='201102,202005'
    time_freq= 'M1'
    
    stats_times='20110205,20200505'
    time_freq= 'D1'
    
    stats_times=['2011,2040','0505,0805']
    time_freq= 'D2'
    
    time_freq = 'M2'
    stats_times = ["2010,2025", "1,2"]