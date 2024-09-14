# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:01:50 2024

@author: EDY

:param element：对应原型，传入的要素名称，可多选
气象站数据：
        平均气温	TEM_Avg 
        最高气温	TEM_Max
        最低气温	TEM_Min
        
        降水量	PRE_Time_2020

        大蒸发	EVP_Big
        小蒸发	EVP
        高桥蒸发	EVP_Taka
        彭曼蒸发	EVP_Penman

        日照时数	SSH
        平均相对湿度	RHU_Avg
        平均风速	WIN_S_2mi_Avg
        10分钟平均最大风速 ？？？？？？？？
        平均地面温度	GST_Avg
        最高地面温度	GST_Max
        最低地面温度	GST_Min
        平均5cm地温 GST_Avg_5cm
        平均10cm地温	GST_Avg_10cm
        平均15cm地温	GST_Avg_15cm
        平均20cm地温	GST_Avg_20cm
        平均40cm地温	GST_Avg_40cm
        平均80cm地温	GST_Avg_80cm
        平均160cm地温	GST_Avg_160cm
        平均320cm地温	GST_Avg_320cm
        
:param time_freq: 对应原型选择数据的时间尺度
    传参：
    年 - 'Y'
    季 - 'Q'
    月(区间) - 'M' 
    日(连续) - 'D'

:param train_time: 对应原型的参考时段
:param verify_time: 对应原型的参考时段
    (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'
    (2)当time_freq选择季Q。下载连续的月数据，处理成季数据（多下两个月数据），提取历年相应的季节数据，传：['%Y,%Y','3,4,5']，其中'3,4,5'为季度对应的月份 
    (4)当time_freq选择月(区间)M。下载连续的月数据，随后提取历年相应的月数据，传参：['%Y,%Y','11,12,1,2'] 前者年份，'11,12,1,2'为连续月份区间
    (5)当time_freq选择日(连续)D。下载连续的日数据，传参：'%Y%m%d,%Y%m%d'
        
:param sta_ids: 传入的站点，多站，传：'52866,52713,52714'

"""
import pandas as pd
from Module02.page_ice.wrapped.func00_data_read_sql import data_read_sql
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 求得区域平均，区域指的是传进来的所有站点
def data_proce(df,processing_methods, additional_method=None):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['年份'] = df['Datetime'].dt.year
    
    if additional_method:
        processing_methods = {col: additional_method for col in processing_methods.keys()}
    else:
        processing_methods = processing_methods
        
    # 动态处理每个分组
    def process_group(group):
        processed = {
            'Station_Id_C': group['Station_Id_C'].iloc[0],
            'Datetime': group['Datetime'].iloc[0]
        }
        for element, method in processing_methods.items():
            if element in group.columns:
                processed[f'{element}_{method}'] = getattr(group[element], method)()
        return pd.Series(processed)
    
    grouped_data = df.groupby(['Station_Id_C', '年份']).apply(process_group).reset_index(drop=True)
    
    # 保存站点数据
    station_data=grouped_data.copy()
    station_data['年'] = station_data['Datetime'].dt.year
    station_data.drop(['Datetime'], axis=1, inplace=True) 

    
    grouped_data['年份'] = grouped_data['Datetime'].dt.year
    average_data = grouped_data.iloc[:,2::].groupby('年份').mean()
    
    return station_data,average_data
    
def factor_data_deal(element,train_time,verify_time,sta_ids,time_freq,time_freq_data,time_freq_main,processing_methods):

    
    # 2. 参数分类
    weather_element=['TEM_Avg','TEM_Max','TEM_Min','PRE_Time_2020','EVP_Big','EVP',
                     'EVP_Taka','EVP_Penman','SSH','RHU_Avg','WIN_S_2mi_Avg','GST_Avg',
                     'GST_Max','GST_Min','GST_Avg_5cm','GST_Avg_10cm','GST_Avg_15cm',
                     'GST_Avg_20cm','GST_Avg_40cm','GST_Avg_80cm','GST_Avg_160cm','GST_Avg_320cm'] 
    

    # 3. 确定表名
    weather_table_name = 'qh_climate_cmadaas_day'
    weather_element_str = 'Station_Id_C,Station_Name,Datetime,'
    
    elements=element.split(',')
    time_freqs =time_freq.split(',')
    
    result_dict = {}

    for element, time_freq in zip(elements, time_freqs):
        if time_freq in result_dict:
            result_dict[time_freq] += ',' + element
        else:
            result_dict[time_freq] = element
    elements = []
    time_freqs = []
    for key, value in result_dict.items():
        elements.append(value)
        time_freqs.append(key)

    # 4. 读取数据
    # 针对每个要素的时间尺度
    train_dataframes = []
    verify_dataframes = []
    train_station_dataframes = []
    verify_station_dataframes = []

    for index, ele in enumerate(elements):
        weather_element_str = 'Station_Id_C,Station_Name,Datetime,'

        # 根据要素选择相应的数据库
        if ele.split(',')[0] in weather_element:
            if element == 'EVP_Penman':
                weather_element_str = weather_element_str+'pmet'
            else:
                weather_element_str = weather_element_str+ele
        
        # 构建选择时间
        if time_freqs[index] == 'Y':
            train_time_use=train_time
            verify_time_use=verify_time
        
        elif time_freqs[index] == 'Q':# ['%Y,%Y','3,4,5']
            train_time_use=[train_time,time_freq_data[index]]
            verify_time_use=[verify_time,time_freq_data[index]]

        elif time_freqs[index] == 'M2':
            train_time_use=[train_time,time_freq_data[index]]
            verify_time_use=[verify_time,time_freq_data[index]]
            
        elif time_freqs[index] == 'D1':
            train_time_use= train_time.split(',')[0]+'0101,'+train_time.split(',')[1]+'1231'
            verify_time_use= verify_time.split(',')[0]+'0101,'+verify_time.split(',')[1]+'1231'
    
        train_data=data_read_sql(sta_ids,weather_element_str,train_time_use,weather_table_name,time_freqs[index])
        verify_data=data_read_sql(sta_ids,weather_element_str,verify_time_use,weather_table_name,time_freqs[index])
        
        for name in ele.split(','):
            train_data.loc[train_data[name] > 1000, [name]] = np.nan
            verify_data.loc[verify_data[name] > 1000, [name]] = np.nan

        
        #  5. 转换成年数据 或保留日数据
        if time_freq_main != 'D':
            if time_freqs[index] != 'D':
                train_station_data,train_data_deal=data_proce(train_data,processing_methods)
                verify_station_data,verify_data_deal=data_proce(verify_data,processing_methods)
            else:
                train_station_data,train_data_deal=data_proce(train_data,processing_methods,additional_method='mean')
                verify_station_data,verify_data_deal=data_proce(verify_data,processing_methods,additional_method='mean')
        else:
            train_station_data=train_data.copy()
            train_station_data.drop([ 'Station_Name'], axis=1,inplace=True)

            train_data['Datetime'] = pd.to_datetime(train_data['Datetime'])
            train_data=train_data.set_index(train_data['Datetime'])
            train_data.drop([ 'Station_Id_C','Station_Name','Datetime'], axis=1,inplace=True)
            train_data_deal = train_data.resample('D').mean()
            
            verify_station_data=verify_data.copy()
            verify_station_data.drop([ 'Station_Name'], axis=1,inplace=True)
            
            verify_data['Datetime'] = pd.to_datetime(verify_data['Datetime'])
            verify_data=verify_data.set_index(verify_data['Datetime'])
            verify_data.drop([ 'Station_Id_C','Station_Name','Datetime'], axis=1,inplace=True)
            verify_data_deal = verify_data.resample('D').mean()

        train_dataframes.append(train_data_deal)
        verify_dataframes.append(verify_data_deal)
        train_station_dataframes.append(train_station_data)
        verify_station_dataframes.append(verify_station_data)

    combined_train = pd.concat(train_dataframes, axis=1)
    combined_verify = pd.concat(verify_dataframes, axis=1)
    

    combined_train_station=train_station_dataframes[0].copy()
    combined_verify_station=verify_station_dataframes[0].copy()
    for i in np.arange(len(train_station_dataframes))[1::]:
        combined_train_station= pd.merge(combined_train_station,train_station_dataframes[i], on=['Station_Id_C', '年'])
        combined_verify_station= pd.merge(combined_verify_station,verify_station_dataframes[i], on=['Station_Id_C', '年'])

    # combined_train=combined_train.reset_index()
    # combined_verify=combined_verify.reset_index()
    # combined_train = combined_train.rename(columns={'年份': 'Datetime'})
    # combined_verify = combined_verify.rename(columns={'年份': 'Datetime'})
    # combined_train = combined_train.set_index(combined_train['Datetime'].astype(str))
    # combined_verify = combined_verify.set_index(combined_verify['Datetime'].astype(str))


    return combined_train_station,combined_verify_station,combined_train,combined_verify
    
    
    

    
#%%
if __name__=='__main__':
    
    element='TEM_Avg,PRE_Time_2020,RHU_Avg,WIN_S_2mi_Avg,SSH'
    train_time='2019,2024'
    verify_time='2019,2022'
    sta_ids='51886,52737,52842,52886,52876'
    time_freq='Y,Q,M,D,Y'
    time_freq_main='Y'
    time_freq_data='0,1,3,0,0'
    train_station,verify_station,train_data,verify_data=factor_data_deal(element,train_time,verify_time,sta_ids,time_freq)
        
        
        
        
        
        
        
        
        
        
        
        
        
   


