import os
import glob
import json
import numpy as np
import pandas as pd
from Utils.config import cfg


def rain_change_processing(data_in):
    '''
    气候变化影响预估-降水-站点数据处理，计算Rx5day和R20mm
    转换到年尺度，计算风险
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
    
    # 要素处理
    if 'PRE_Time_2020' in df_data.columns:
        df_data['PRE_Time_2020'] = df_data['PRE_Time_2020'].map(str)
        df_data.loc[df_data['PRE_Time_2020'].str.contains('9999'), 'PRE_Time_2020'] = np.nan
        df_data['PRE_Time_2020'] = df_data['PRE_Time_2020'].map(float)

    def sample(x):
        x_info = x[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Alti']].resample('1A').first()
        rx5day = x['PRE_Time_2020'].resample('1M').apply(lambda x: x.rolling(5).sum().max())
        rx5day = rx5day.resample('1A').max()
        rx5day = (rx5day-rx5day.min())/(rx5day.max()-rx5day.min()) # 0-1标准化
        rx5day.fillna(0, inplace=True)

        x['r20'] = np.where(x['PRE_Time_2020']>=20, 1, 0)
        r20 = x['r20'].resample('1A').sum()
        r20 = (r20-r20.min())/(r20.max()-r20.min()) # 0-1标准化
        r20.fillna(0, inplace=True)

        concat = pd.concat([x_info, rx5day, r20], axis=1)
        return concat

    # 各个站点的历年rx5day
    df_out = df_data.groupby('Station_Id_C').apply(sample)
    df_out.reset_index(level=0,inplace=True,drop=True)
    df_out.columns = ['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Alti', 'rx5day', 'r20']
    df_out['Alti'] = 100/(df_out['Alti']+145)

    return df_out