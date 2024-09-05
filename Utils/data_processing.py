import os
import glob
import json
import numpy as np
import pandas as pd
from Utils.config import cfg


def wind_direction_to_symbol(x):
    '''
    pandas的apply函数
    把任何形式的风向统一处理为字母符号形式，年/月/日数据
    '''
    try:
        x = float(x)
        if (348.76 <= x <= 360.0) or (0 <= x <= 11.25) or (x == 999001):
            x = 'N'

        elif (11.26 <= x <= 33.75) or (x == 999002):
            x = 'NNE'

        elif (33.76 <= x <= 56.25) or (x == 999003):
            x = 'NE'

        elif (56.26 <= x <= 78.75) or (x == 999004):
            x = 'ENE'

        elif (78.76 <= x <= 101.25) or (x == 999005):
            x = 'E'

        elif (101.26 <= x <= 123.75) or (x == 999006):
            x = 'ESE'

        elif (123.76 <= x <= 146.25) or (x == 999007):
            x = 'SE'

        elif (146.26 <= x <= 168.75) or (x == 999008):
            x = 'SSE'

        elif (168.26 <= x <= 191.75) or (x == 999009):
            x = 'S'

        elif (191.26 <= x <= 213.75) or (x == 999010):
            x = 'SSW'

        elif (213.26 <= x <= 236.75) or (x == 999011):
            x = 'SW'

        elif (236.26 <= x <= 258.75) or (x == 999012):
            x = 'WSW'

        elif (258.26 <= x <= 281.75) or (x == 999013):
            x = 'W'

        elif (281.26 <= x <= 303.75) or (x == 999014):
            x = 'WNW'

        elif (303.26 <= x <= 326.75) or (x == 999015):
            x = 'NW'

        elif (326.26 <= x <= 348.75) or (x == 999016):
            x = 'NNW'

        elif x == 999017:
            x = 'C'
        # 异常值
        elif x in [999999, 999982, 999983]:
            x = np.nan

    except:
        x = x.upper()

    return x

def data_processing(data_in, element, degree=None):
    '''
    年/月/日数据前处理
    月（季）数据和日数据，都转换为年尺度
    element: 指定的要素
    '''
    if data_in is None or data_in.empty:
        return data_in
    df_data = data_in.copy()
    df_data.set_index('Datetime', inplace=True)
    df_data.index = pd.DatetimeIndex(df_data.index)
    df_data['Station_Id_C'] = df_data['Station_Id_C'].astype(str)

    if 'Unnamed: 0' in df_data.columns:
        df_data.drop(['Unnamed: 0'], axis=1, inplace=True)

    # 1.首先进行要素处理
    if 'PRE_Time_2020' in df_data.columns:
        df_data['PRE_Time_2020'] = df_data['PRE_Time_2020'].map(str)
        df_data.loc[df_data['PRE_Time_2020'].str.contains('9999'), 'PRE_Time_2020'] = np.nan
        df_data['PRE_Time_2020'] = df_data['PRE_Time_2020'].map(float)

    if 'PRE_Max_Day' in df_data.columns:
        df_data['PRE_Max_Day'] = df_data['PRE_Max_Day'].apply(lambda x: np.nan if x > 999 else x)

    if 'SSP_Mon' in df_data.columns:
        df_data['SSP_Mon'] = df_data['SSP_Mon'].apply(lambda x: np.nan if x > 999 else x)
    
    if 'WIN_D_S_Max_C' in df_data.columns:
        df_data['WIN_D_S_Max_C'] = df_data['WIN_D_S_Max_C'].astype(str).apply(wind_direction_to_symbol)

    if 'WIN_D_Max_C' in df_data.columns:
        df_data['WIN_D_Max_C'] = df_data['WIN_D_Max_C'].astype(str).apply(wind_direction_to_symbol)
    
    if 'Cov' in df_data.columns: # 草地覆盖度
        df_data['Cov'] = df_data['Cov'].apply(lambda x: np.nan if x > 999 else x)

    # 计算积温
    if degree is not None:
        assert element == 'TEM_Avg', '计算积温要素错误，不是日平均气温'
        df_data['TEM_Avg'] = np.where(df_data['TEM_Avg']<degree,np.nan,df_data['TEM_Avg'])
        df_data.rename(columns={'TEM_Avg': 'Accum_Tem'}, inplace=True)
        element = 'Accum_Tem'

    # 2.时间转换
    resample_max = ['TEM_Max', 'PRS_Max', 'WIN_S_Max', 'WIN_S_Inst_Max', 'GST_Max', 'huangku']
    resample_min = ['TEM_Min', 'PRS_Min', 'GST_Min', 'RHU_Min', 'fanqing']
    resample_sum = ['PRE_Time_2020', 'PRE_Days', 'EVP_Big', 'EVP', 'EVP_Taka', 'PMET','sa','rainstorm','light_snow','snow',
                    'medium_snow','heavy_snow','severe_snow','Hail_Days','Hail','GaWIN',
                    'GaWIN_Days','SaSt','SaSt_Days','FlSa','FlSa_Days','FlDu','FlDu_Days',
                    'Thund','Thund_Days''high_tem','drought','light_drought','medium_drought',
                    'heavy_drought','severe_drought','Accum_Tem']
    
    resample_mean = ['TEM_Avg', 'PRS_Avg', 'WIN_S_2mi_Avg', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Avg_5cm', 'GST_Avg_10cm', 
                     'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 
                     'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 'SSH', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg', 'Cov', 'pmet','EVP_Taka']

    def sample(x):
        '''
        重采样的applyfunc
        '''
        x_info = x[['Station_Id_C', 'Station_Name', 'Lat', 'Lon']].resample('1A').first()
        if element in resample_max:
            x_res = x[element].resample('1A').max()
        elif element in resample_min:
            x_res = x[element].resample('1A').min()
        elif element in resample_sum:
            x_res = x[element].resample('1A').sum()
        elif element in resample_mean:
            x_res = x[element].resample('1A').mean().round(1)

        x_concat = pd.concat([x_info, x_res], axis=1)
        return x_concat
    
    df_data = df_data.groupby('Station_Id_C').apply(sample)  # 月数据和日数据转换为1年一个值
    df_data = df_data.replace(to_replace='None', value=np.nan).dropna()
    df_data.reset_index(level=0, drop=True, inplace=True)
    
    return df_data