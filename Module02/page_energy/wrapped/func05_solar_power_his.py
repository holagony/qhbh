# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:41:03 2024

@author: EDY

能源影响预估 -- 太阳能

:param element: 
    总辐射量：TR 
    直接辐射占比： PDR;
    日照时数：SH；
    有效日照天数：ASD
"""
import pandas as pd
import numpy as np
from Utils.data_processing import data_processing
import calendar

def radiation_partition(df,lon,lat):
    '''
    晴空指数法
    将总辐射数据划分为直接辐射和散射辐射
    '''
    # 创建EQ时差表
    table = pd.DataFrame()
    table['平年'] = list(range(1,32)) + [np.nan]
    table['闰年'] = [np.nan] + list(range(1,32))
    table[1] = [-2,-3,-3,-4,-4,-5,-5,-5,-6,-6,-7,-7,-7,-8,-8,-9,-9,-9,-10,-10,-10,-11,-11,-11,-11,-12,-12,-12,-12,-13,-13,np.nan]
    table[2] = [-13,-13,-13,-13,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-13,-13,-13,-13,np.nan,np.nan,np.nan]
    table[3] = [-13,-13,-13,-12,-12,-12,-12,-12,-11,-11,-11,-11,-10,-10,-10,-10,-9,-9,-9,-8,-8,-8,-8,-7,-7,-7,-6,-6,-6,-5,-5,-5]
    table[4] = [-5,-4,-4,-4,-3,-3,-3,-3,-2,-2,-2,-1,-1,-1,-1,-0,-0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,np.nan]
    table[5] = [3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3]
    table[6] = [3,2,2,2,2,2,2,1,1,1,1,1,1,0,0,-0,-0,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2,-3,-3,-3,-3,np.nan]
    table[7] = [-3,-4,-4,-4,-4,-4,-4,-5,-5,-5,-5,-5,-5,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-7,-7,-7,-7,-7,-7,-7,-7,-7]
    table[8] = [-7,-7,-7,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-4,-4,-3,-3,-3,-3,-2,-2,-2,-1,-1,-1]
    table[9] = [-1,-0,-0,0,1,1,1,2,2,2,3,3,3,4,4,5,5,5,6,6,6,7,7,8,8,8,9,9,10,10,10,np.nan]
    table[10] = [10,10,11,11,11,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16]
    table[11] = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,15,15,15,15,15,14,14,14,14,13,13,13,12,12,12,11,11,np.nan]
    table[12] = [11,11,10,10,10,9,9,8,8,8,7,7,6,6,5,5,5,4,4,3,3,2,2,1,1,0,-0,-1,-1,-1,-2,-2]
    
    n = df.index.day_of_year.values # 积日
    EDNI = 1366.1*(1+0.033*np.cos(360*n/365))
    phi = lat
    delta = 23.45*np.sin(360*(284+n)/365)
    c_t = df.index.hour.values
    l_g = lon
    l_c = 4*(l_g-120)/60

    def query_eq(x):
        '''
        EQ表查表确定分钟值，转换为小时
        '''

        if calendar.isleap(x['Year']):
            row = table.loc[table['闰年']==x['Day'],table.columns==x['Mon']]
        
        else:
            row = table.loc[table['平年']==x['Day'],table.columns==x['Mon']]
        
        return row.values[0][0]/60

    # e_q = df.apply(query_eq).values
    e_q_values = []
    for index, row in df.iterrows():
        e_q_value = query_eq(row)
        e_q_values.append(e_q_value)

    # 将结果转换为数组
    e_q = np.array(e_q_values)
    
    t_t = c_t + float(l_c) + e_q
    omega = (t_t-12)*15
    EHI = EDNI*(np.cos(float(phi))*np.cos(delta)*np.cos(omega)+np.sin(float(phi))*np.sin(delta))

    kt = df['v14311'].astype(float)/EHI # 晴空系数
    kt = np.abs(kt)
    
    def calc_f_kt(x):
        if 0<=x<0.35:
            return 1.0-0.249*x
        elif 0.35<=x<=0.75:
            return 1.557-1.84*x
        elif x>0.75:
            return 0.177
        
    f_kt = kt.apply(calc_f_kt) # 比例
    df['散射辐射辐照量'] = (df['v14311'].astype(float)*f_kt).round(2)
    df['直接辐射辐照量'] = df['v14311'].astype(float) - df['散射辐射辐照量']
    
    return df

def energy_solar_his(element,df):
# 识别站点和年份

    if element=='TR':
        # 1.总辐射量 
        df['v14311'][ df['v14311']==999999]=np.nan
        df['v14311']=df['v14311'].astype(float)*3600 / 1e6 
        df=data_processing(df,'v14311', degree=None)
        df = df.pivot_table(index=df.index, columns=['Station_Id_C'], values='v14311')  # 参考时段df

        df.index = df.index.strftime('%Y')
        df.reset_index(inplace=True)
        df.columns.values[0] = '年'
        result=df.copy()
        
    elif element=='PDR':

        # 2.直接辐射占比
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        except:
            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H')
            
        df.set_index('Datetime', inplace=True)
        df['Station_Id_C'] = df['Station_Id_C'].astype(str)
      
        df["Year"] = df["Year"].astype(int)
        df["Mon"] = df["Mon"].astype(int)
        df["Day"] = df["Day"].astype(int)
        df["Lon"] = df["Lon"].astype(float)
        df["Lat"] = df["Lat"].astype(float)

        station=df['Station_Id_C'].unique()
        result=[]
        for i in station:
            station_c=df[df['Station_Id_C']==i]
            lon=station_c['Lon'].iloc[0]
            lat=station_c['Lat'].iloc[0]
            # break
            result.append(radiation_partition(station_c,lon,lat))
        
        result= pd.concat(result)
        result['直接辐射占比']=(result['直接辐射辐照量'].astype(float)/result['v14311'].astype(float))*100
      
        
        result = result.pivot_table(index=result.index, columns=['Station_Id_C'], values='直接辐射占比')  # 参考时段df
        result = result.resample('Y').mean().astype(float).round(2)
        result.index = result.index.strftime('%Y')
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
    
    elif element=='SH':

        # 3.日照时数
        df=data_processing(df,'ssh', degree=None)
        result = df.pivot_table(index=df.index, columns=['Station_Id_C'], values='ssh')  # 参考时段df

        result.index = result.index.strftime('%Y')
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
        
    elif element=='ASD':

        # 4. 有效日照天数
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        except:
            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H')
            
        df.set_index('Datetime', inplace=True)
        df['Station_Id_C'] = df['Station_Id_C'].astype(str)

        df['Lon'] = df['Lon'].astype(float)
        df['Lat'] = df['Lat'].astype(float)
        df['ssh'] = df['ssh'].astype(float)
        
        def sample(x):
            '''
            重采样的applyfunc
            '''
            x_info = x[['Station_Id_C', 'Station_Name', 'Lat', 'Lon']].resample('1D').first()
            x_res = x['ssh'].resample('1D').sum()
            x_concat = pd.concat([x_info, x_res], axis=1)
            return x_concat
        
        df = df.groupby('Station_Id_C').apply(sample)  # 月数据和日数据转换为1年一个值
        df = df.replace(to_replace='None', value=np.nan).dropna()
        df.reset_index(level=0, drop=True, inplace=True)
        df['ASD']=(df['ssh']>=3).astype(int)
        result = df.pivot_table(index=df.index, columns=['Station_Id_C'], values='ASD')  
        result = result.resample('Y').sum().astype(int)
        result.index = result.index.strftime('%Y')
        result.reset_index(inplace=True)
        result.columns.values[0] = '年'
    return result











