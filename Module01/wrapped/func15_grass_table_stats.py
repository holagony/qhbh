# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:40:04 2024

@author: EDY
"""

from Utils.data_loader_with_threads import get_database_result
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def grass_table_stats(element,refer_years,nearly_years,time_freq,stats_times,sta_ids):
    
    elements= 'Datetime,Station_Id_C,Station_Name,Lon,Lat,value,type'
    sta_ids = tuple(sta_ids.split(','))

    if element=='grassland_green_period':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(11,),1,1)
        refer_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(11,),0,1)
        nearly_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(11,),0,1)
        data_r_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(10,),0,0)

        station_df.reset_index(inplace=True,drop=True)
        last_df=nearly_df.iloc[-1:].copy()
        
    elif element=='grassland_yellow_period':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(21,),1,1)
        refer_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(21,),0,1)
        nearly_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(21,),0,1)
        data_r_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(20,),0,0)

        station_df.reset_index(inplace=True,drop=True)
        last_df=nearly_df.iloc[-1:].copy()
        
    elif element=='grassland_growth_period':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(31,),1,1)
        refer_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(31,),0,1)
        nearly_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(31,),0,1)

        station_df.reset_index(inplace=True,drop=True)
        last_df=nearly_df.iloc[-1:].copy()
        
    elif element=='grassland_yield':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(40,),1,1)
        refer_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(40,),0,1)
        nearly_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(40,),0,1)

        station_df.reset_index(inplace=True,drop=True)
        last_df=nearly_df.iloc[-1:].copy()
    
    elif element=='grassland_coverage':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(50,),1,1)
        refer_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(50,),0,1)
        nearly_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(50,),0,1)

        station_df.reset_index(inplace=True,drop=True)
        last_df=nearly_df.iloc[-1:].copy()
    
    elif element=='grass_height':

        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(60,),1,1)
        refer_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(60,),0,1)
        nearly_df=get_database_result(sta_ids, elements, 'qh_climate_agro_customer', time_freq, stats_times,(60,),0,1)

        station_df.reset_index(inplace=True,drop=True)
        last_df=nearly_df.iloc[-1:].copy()

        

    def trend_rate(x):
        '''
        计算变率（气候倾向率）的pandas apply func
        '''
        try:
            x = x.to_frame()
            x['num'] = np.arange(len(x))
            x.dropna(how='any', inplace=True)
            train_x = x.iloc[:, -1].values.reshape(-1, 1)
            train_y = x.iloc[:, 0].values.reshape(-1, 1)
            model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
            weight = model.coef_[0][0].round(3) * 10
            return weight
        except:
            return np.nan

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df.apply(trend_rate, axis=0).round(1)
    tmp_df.loc['最大值'] = data_df.iloc[:, :].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[:, :].min(axis=0)
    tmp_df.loc['与上一年比较值'] = (data_df.iloc[:, :].mean(axis=0) - last_df.iloc[:, :].mean(axis=0)).round(1)
    tmp_df.loc['近10年均值'] = nearly_df.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['与近10年比较值'] = (data_df.iloc[:, :].mean(axis=0) - nearly_df.iloc[:, :].mean(axis=0)).round(1)
    tmp_df.loc['参考时段均值'] = refer_df.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']).round(1)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'] / tmp_df.loc['参考时段均值']) * 100).round(2)

    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).round(1)
    stats_result['区域距平'] = (stats_result.iloc[:, :].mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).round(1)
    stats_result['区域距平百分率'] = (stats_result['区域距平']/refer_df.iloc[:, :].mean().mean()).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:, :-3].max(axis=1)
    stats_result['区域最小值'] = stats_result.iloc[:, :-4].min(axis=1)
    
    # 在concat前增加回归方程
    def lr(x):
        try:
            x = x.to_frame()
            x['num'] = x.index.tolist()
            x['num'] = x['num'].map(int)
            x.dropna(how='any', inplace=True)
            train_x = x.iloc[:, -1].values.reshape(-1, 1)
            train_y = x.iloc[:, 0].values.reshape(-1, 1)
            model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
            weight = model.coef_[0][0].round(3)
            bias = model.intercept_[0].round(3)
            return weight, bias
        except:
            return np.nan, np.nan
   
    reg_params = pd.DataFrame()
    reg_params = stats_result.apply(lr, axis=0)
    reg_params = reg_params.T
    reg_params.reset_index(drop=False,inplace=True)
    reg_params.columns = ['站号','weight','bias']
    
    # concat
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    

    stats_result.reset_index(drop=True, inplace=True)
    post_data_df = data_df.copy()
    post_refer_df = refer_df.copy()
    
    if element in ['grassland_green_period','grassland_yellow_period']:
        return stats_result, post_data_df, post_refer_df, reg_params,station_df,data_r_df
    else:
        return stats_result, post_data_df, post_refer_df, reg_params,station_df

        
        
        
    