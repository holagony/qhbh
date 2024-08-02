import os
import glob
import json
import numpy as np
import pandas as pd
from Utils.config import cfg
from Utils.cost_time import cost_time


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

def data_processing(yearly_data):
    '''
    年/月/日数据前处理
    '''
    if yearly_data is None or yearly_data.empty:
        return yearly_data
    df_data = yearly_data.copy()
    df_data.set_index('Datetime', inplace=True)
    df_data.index = pd.DatetimeIndex(df_data.index)
    df_data['Station_Id_C'] = df_data['Station_Id_C'].astype(str)

    if 'Unnamed: 0' in df_data.columns:
        df_data.drop(['Unnamed: 0'], axis=1, inplace=True)

    # 根据要素处理
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

    return df_data