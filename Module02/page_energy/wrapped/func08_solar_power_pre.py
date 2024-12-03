# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:21:24 2024

@author: EDY
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Module02.page_energy.wrapped.func06_read_model_data import read_model_data


# 系数和站

def a_b_statiuon(station_id):
    
    result=pd.DataFrame(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])
    if station_id in ["52754","52853" , "52765" , "52657","52633" , "52645", "52745" ,"52833" ,"52842" , "52856"]:
        result.loc[0]=[0.46 ,0.539,0.653,0.403,0.371,0.405,0.368,0.426,0.325,0.456,0.345,0.504]
        result.loc[1]=[0.005,0.004,0.003,0.006,0.006,0.004,0.005,0.005,0.006,0.005,0.006,0.005]

    elif station_id in ["52818","51886","52602","52713","52707","52836","52737","52825"]:
        result.loc[0]=[0.319,0.268,0.431,0.368,0.355,0.3  ,0.258,0.354,0.358,0.284,0.168,0.267]
        result.loc[1]=[0.007,0.007,0.006,0.006,0.006,0.007,0.008,0.006,0.006,0.007,0.008,0.008 ]

    elif station_id in ["52866","52862","52863","52855","52869","52875","52874","52876","52877","52972","52963","52868","52974"]:
        result.loc[0]=[0.393,0.336,0.382,0.387,0.344,0.26 ,0.286,0.312,0.274,0.293,0.307,0.276]
        result.loc[1]=[0.005,0.005,0.006,0.005,0.005,0.006,0.006,0.006,0.006,0.006,0.006,0.007 ]

    elif station_id in ["56029","56018","56016","56021","56034","56125","56004","52908"]:
        result.loc[0]=[0.349,0.329,0.279,0.314,0.423,0.376,0.285,0.396,0.3  ,0.494,0.251,0.165]
        result.loc[1]=[0.007,0.006,0.008,0.006,0.004,0.004,0.007,0.005,0.006,0.003,0.008,0.01]

    elif station_id in ["56043","56045","56046","56067","56151","56033","56065","52968","52957","52955"]:
        result.loc[0]=[0.228,0.401,0.448,0.349,0.208,0.317,0.334,0.312,0.376,0.185,0.787,0.83]
        result.loc[1]=[0.009,0.005,0.006,0.007,0.009,0.006,0.006,0.007,0.005,0.009,0.001,0.001]
    else:
        result.loc[0]=[0.228,0.401,0.448,0.349,0.208,0.317,0.334,0.312,0.376,0.185,0.787,0.83]
        result.loc[1]=[0.009,0.005,0.006,0.007,0.009,0.006,0.006,0.007,0.005,0.009,0.001,0.001]
        
    return result

def solar_power_pre(element,data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids,station_dict):
    
    df=read_model_data(data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_ids)
    df=df/3600


    if element=='SH':

        # 3.日照时数
        result=df.resample('Y').sum()

        result.index = result.index.strftime('%Y')
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
        
    elif element=='ASD':

        # 4. 有效日照天数
    
        df_asd=(df>=3).astype(int)
        result = df_asd.resample('Y').sum().astype(int)
        result.index = result.index.strftime('%Y')
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
    
    elif element=='TR':
    
        df_a=df.copy()
        df_b=df.copy()
        for index, row in df.iterrows():
            month = index.month
            row=row.to_frame().T
            
            for column in row.columns:
                df_a.at[index,column]=a_b_statiuon(column).at[0,str(month)]
                df_b.at[index,column]=a_b_statiuon(column).at[1,str(month)]
                
        result=df_a*df+df_b
        result = result.resample('Y').sum().round(2)
        result.index = result.index.strftime('%Y')
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
    
    return result
    
    
    
    
    
    
    


    