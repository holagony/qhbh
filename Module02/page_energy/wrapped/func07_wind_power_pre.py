# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:35:13 2024

@author: EDY
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Module02.page_energy.wrapped.func06_read_model_data import read_model_data
from Utils.data_processing import wind_direction_to_symbol


def wind_power_pre(element,data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids,station_dict):
    
    if element=='WSF':

        # 风速频率
        df=read_model_data(data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids)

        station_names =df.columns
        years=df.index.year.unique()
        
        ws_bins = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21.5]  
        ws_bins_cols = [str(ws_bins[i - 1]) + '-' + str(ws_bins[i]) for i in range(1, len(ws_bins))]  
        
        result=dict()
        for station_name in station_names:
            result[station_name]=dict()
        
            data=df[station_name].copy()
            i=0
            for year in years: 
                # break
                data_year=data[data.index.year==year].to_frame()
                ws_hist, _ = np.histogram(data_year[station_name], bins=ws_bins)  
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
            
    elif element=='WDF':
        
        df=read_model_data(data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids)

        station_names =df.columns
        years=df.index.year.unique()
       
        result=dict()
        for station_name in station_names:
            result[station_name]=dict()
        
            data=df[station_name].copy()
            i=0
            for year in years:
                # break
                data_year=data[data.index.year==year].to_frame()
                data_year = data_year[station_name].astype(str).apply(wind_direction_to_symbol).to_frame()
                counts = data_year[station_name].value_counts().to_frame()
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
            
    elif element=='WPD':

        df_psl=read_model_data(data_dir,time_scale,insti,scene,'psl',stats_times,time_freq,station_ids)
        df_tem=read_model_data(data_dir,time_scale,insti,scene,'tas',stats_times,time_freq,station_ids)
        df_ws=read_model_data(data_dir,time_scale,insti,scene,'ws',stats_times,time_freq,station_ids)

        station_names =df_ws.columns
        years=df_ws.index.year.unique()
        
        input_ws=3
        output_ws=25
        

        result=pd.DataFrame(columns=station_names)
        for station_name in station_names:
            result[station_name]=dict()
        
            df_psl_sta=df_psl[station_name].copy().to_frame()
            df_tem_sta=df_tem[station_name].copy().to_frame()
            df_ws_sta=df_ws[station_name].copy().to_frame()

            
            # 先求空气密度
            rho = (df_psl_sta[station_name].astype(float) * 100 / (287 * (df_tem_sta[station_name].astype(float) + 273.15))).mean()  
            mask_ws = df_ws_sta[station_name].astype(float).to_frame().mask((df_ws_sta[station_name].astype(float).to_frame() > output_ws) | (df_ws_sta[station_name].astype(float).to_frame() < input_ws), np.nan)
            mask_pd = (1 / 2) * rho * (mask_ws**3)
            mask_pd_yearly = mask_pd.resample('1Y').mean().round(1)
            
            result[station_name]=  mask_pd_yearly
            
        result.index = result.index.strftime('%Y')
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
        result=result
    
    return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    return result_days,result_hdd18,result_start_end,result_start_end_num


    