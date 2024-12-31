import os
import glob
import json
import numpy as np
import pandas as pd
from Utils.config import cfg


def station_traffic_processing(data_in, element_str):
    '''
    年/月/日数据前处理
    月（季）数据和日数据，都转换为年尺度
    element: 指定的要素
    '''
    if data_in is None or data_in.empty:
        return data_in
    df_data = data_in.copy()
    
    try:
        df_data['Datetime'] = pd.to_datetime(df_data['Datetime'])
    except:
        df_data['Datetime'] = pd.to_datetime(df_data['Datetime'], format='%Y%m%d%H')
        
    df_data.set_index('Datetime', inplace=True)
    df_data['Station_Id_C'] = df_data['Station_Id_C'].astype(str)

    df_data['Lon'] = df_data['Lon'].astype(float)
    df_data['Lat'] = df_data['Lat'].astype(float)
    
    element_str = element_str.split(',')
    for element in element_str:
        df_data[element] = df_data[element].astype(float)

    # 1.首先进行要素处理
    if 'PRE_Time_2020' in df_data.columns:
        df_data['PRE_Time_2020'] = df_data['PRE_Time_2020'].map(str)
        df_data.loc[df_data['PRE_Time_2020'].str.contains('9999'), 'PRE_Time_2020'] = np.nan
        df_data['PRE_Time_2020'] = df_data['PRE_Time_2020'].map(float)

    if 'TEM_Avg' in df_data.columns:
        df_data['TEM_Avg'] = df_data['TEM_Avg'].apply(lambda x: np.nan if x > 999 else x)
    
    if 'WIN_S_2mi_Avg' in df_data.columns: # 草地覆盖度
        df_data['WIN_S_2mi_Avg'] = df_data['WIN_S_2mi_Avg'].apply(lambda x: np.nan if x > 999 else x)
    
    
    # 生成flag，判断这一天是不是交通不利日
    # 降水50mm 气温0-35度以外 风大于15m/s，占一个，就是不利日
    df_data['pre_flag'] = np.where((df_data['PRE_Time_2020']<50) | (pd.isna(df_data['PRE_Time_2020'])), 0 ,1)
    df_data['tem_flag'] = np.where(((df_data['TEM_Avg']>0) & (df_data['TEM_Avg']<35)) | (pd.isna(df_data['TEM_Avg'])),0,1)
    df_data['win_flag'] = np.where((df_data['WIN_S_2mi_Avg']<15) | (pd.isna(df_data['WIN_S_2mi_Avg'])),0,1)
    df_data['traffic'] = df_data.loc[:,['pre_flag','tem_flag','win_flag']].max(axis=1)
    df_data = df_data[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'traffic']]

    # 2.时间转换为年
    def sample(x):
        '''
        重采样的applyfunc
        '''
        x_info = x[['Station_Id_C', 'Station_Name', 'Lat', 'Lon']].resample('1A').first()
        x_res = x['traffic'].resample('1A').sum()
        x_concat = pd.concat([x_info, x_res], axis=1)
        return x_concat
    
    df_data = df_data.groupby('Station_Id_C').apply(sample)  # 月数据和日数据转换为1年一个值
    df_data = df_data.replace(to_replace='None', value=np.nan).dropna()
    df_data.reset_index(level=0, drop=True, inplace=True)
    
    return df_data