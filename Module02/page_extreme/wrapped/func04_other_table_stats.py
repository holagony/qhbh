# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:16:38 2024

@author: EDY
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing


def other_table_stats(data_df, time_freq, ele):
    '''
    data_df 天擎统计时段数据
    refer_df 天擎参考时段数据
    nearly_df 天擎近10年数据
    time_freq 数据的时间类型 年/月/季/小时
    ele 计算的要素
    last_year 近1年年份
    '''
    ele_dict=dict()
    ele_dict['Hail']='Hail_Days'
    ele_dict['GaWIN']='GaWIN_Days'
    ele_dict['sa']='sa'
    ele_dict['SaSt']='SaSt_Days'
    ele_dict['FlSa']='FlSa_Days'
    ele_dict['FlDu']='FlDu_Days'
    ele_dict['rainstorm']='rainstorm'
    ele_dict['snow']='snow'
    ele_dict['light_snow']='light_snow'
    ele_dict['medium_snow']='medium_snow'
    ele_dict['heavy_snow']='heavy_snow'
    ele_dict['severe_snow']='severe_snow'
    ele_dict['high_tem']='high_tem'
    ele_dict['Thund']='Thund_Days'
    ele_dict['drought']='drought'
    ele_dict['light_drought']='light_drought'
    ele_dict['medium_drought']='medium_drought'
    ele_dict['heavy_drought']='heavy_drought'
    ele_dict['severe_drought']='severe_drought'
    data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values=ele_dict[ele]) # 统计时段df
    data_df = data_df.round(1)

    # if time_freq in ['Y','Q']:
    data_df = data_df.resample('Y').sum()
    
    data_df.index = data_df.index.strftime('%Y')
    data_df.reset_index(inplace=True)
    data_df.rename(columns={'Datetime': '年'}, inplace=True)
    data_df['年'] = data_df['年'].astype('int64')
    # elif time_freq in ['M1','M2']:
    #     data_df.index = data_df.index.strftime('%Y-%m')
    #     refer_df.index = refer_df.index.strftime('%Y-%m')
    #     nearly_df.index = nearly_df.index.strftime('%Y-%m')
    #     last_df.index = last_df.index.strftime('%Y-%m')


    
    return data_df

