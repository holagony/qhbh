# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:08:24 2024

@author: EDY

能源影响预估 -- 风能

有效风功率预估
风向频率
风速频率
有效小时数 : 统计出代表年测风序列中风速在3－25m/s之间的累计小时数。
"""
import pandas as pd
import numpy as np
from Utils.data_processing import wind_direction_to_symbol


def energy_wind_power_his(element,df):
# 识别站点和年份
    station_names =df['Station_Name'].unique()
    years=df['Year'].unique()
    
    if element=='WDF':
        # 1.风向频率 
        result=dict()
        for station_name in station_names:
            result[station_name]=dict()
        
            data=df[df['Station_Name']==station_name].copy()
            i=0
            for year in years:
                # break
                data_year=data[data['Year']==year]
                data_year = data_year['WIN_D_Avg10mi'].astype(str).apply(wind_direction_to_symbol).to_frame()
                counts = data_year['WIN_D_Avg10mi'].value_counts().to_frame()
                freq = ((counts / counts.sum()) * 100).round(1)
                
                if i == 0:
                    all_freq_wd = freq
                    i=i+1
                else:
                    all_freq_wd = pd.concat([all_freq_wd, freq], axis=1)
            
            all_freq_wd = all_freq_wd.T
            all_freq_wd.insert(loc=0, column='年', value=years)
            all_freq_wd.reset_index(drop=True, inplace=True)
            all_freq_wd.reset_index(inplace=True,drop=True)

            result[station_name]=all_freq_wd.to_dict(orient='records')
            
    elif element=='WSF':

        # 2.风速频率
        ws_bins = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21.5]  
        ws_bins_cols = [str(ws_bins[i - 1]) + '-' + str(ws_bins[i]) for i in range(1, len(ws_bins))]  
        
        result=dict()
        for station_name in station_names:
            result[station_name]=dict()
        
            data=df[df['Station_Name']==station_name].copy()
            i=0
            for year in years: 
                # break
                data_year=data[data['Year']==year]
                ws_hist, _ = np.histogram(data_year['WIN_S_Avg10mi'], bins=ws_bins)  
                ws_hist = ((ws_hist / ws_hist.sum()) * 100).round(1) 
                ws_hist = pd.DataFrame(ws_hist)
            
                if i == 0:
                    all_ws_hist = ws_hist
                    i=i+1
                else:
                    all_ws_hist = pd.concat([all_ws_hist, ws_hist], axis=1)
        
            all_ws_hist=all_ws_hist.T
            all_ws_hist.insert(loc=0, column='年', value=years)
            all_ws_hist.columns = ['年'] + ws_bins_cols
            all_ws_hist.reset_index(inplace=True,drop=True)
            result[station_name]=all_ws_hist.to_dict(orient='records')
    
    elif element=='AH':

        # 3.逐年有效风小时数
        input_ws=3
        output_ws=25
        result=pd.DataFrame(columns=station_names)
        
        df["Year"] = df["Year"].astype(int)
        df["Mon"] = df["Mon"].astype(int)
        df["Day"] = df["Day"].astype(int)
        df["Hour"] = df["Hour"].astype(int)
            
        for station_name in station_names:
            result[station_name]=dict()
        
            data=df[df['Station_Name']==station_name].copy()
            
            periods = pd.PeriodIndex(year=data["Year"], month=data["Mon"], day=data["Day"], hour=data['Hour'],freq="H")
            data2 = data['WIN_S_Avg10mi'].to_frame().set_index(periods)
            
            ws_hours = data2.resample('1Y').apply(lambda x: len(x[(x >= input_ws) & (x <= output_ws)]))
            result[station_name]=  ws_hours
            
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
        result=result
        
    elif element=='WPD':

        # 4. 有效风功率密度
        input_ws=3
        output_ws=25
        
        df["Year"] = df["Year"].astype(int)
        df["Mon"] = df["Mon"].astype(int)
        df["Day"] = df["Day"].astype(int)
        df["Hour"] = df["Hour"].astype(int)
        
        result=pd.DataFrame(columns=station_names)
        for station_name in station_names:
            result[station_name]=dict()
        
            data=df[df['Station_Name']==station_name].copy()
            periods = pd.PeriodIndex(year=data["Year"], month=data["Mon"], day=data["Day"], hour=data['Hour'],freq="H")
            data2 = data.set_index(periods)
            
            # 先求空气密度
            rho = (data2['PRS'].astype(float) * 100 / (287 * (data2['TEM'].astype(float) + 273.15))).mean()  
            mask_ws = data2['WIN_S_Avg10mi'].astype(float).to_frame().mask((data2['WIN_S_Avg10mi'].astype(float).to_frame() > output_ws) | (data2['WIN_S_Avg10mi'].astype(float).to_frame() < input_ws), np.nan)
            mask_pd = (1 / 2) * rho * (mask_ws**3)
            mask_pd_yearly = mask_pd.resample('1Y').mean().round(1)
            result[station_name]=  mask_pd_yearly
            
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
        result=result

    return result











