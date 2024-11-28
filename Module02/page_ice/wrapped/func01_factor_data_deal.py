# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:01:50 2024

@author: EDY

"""
import pandas as pd
from Module02.page_ice.wrapped.func00_data_read_sql import data_read_sql
import numpy as np
import re

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
                processed[f'{element}'] = getattr(group[element], method)()
        return pd.Series(processed)
    
    grouped_data = df.groupby(['Station_Id_C', '年份']).apply(process_group).reset_index(drop=True)
    
    # 保存站点数据
    station_data=grouped_data.copy()
    station_data['年'] = station_data['Datetime'].dt.year
    station_data.drop(['Datetime'], axis=1, inplace=True) 

    
    grouped_data['年份'] = grouped_data['Datetime'].dt.year
    average_data = grouped_data.iloc[:,2::].groupby('年份').mean()
    
    return station_data,average_data

def clean_column_name(name):
    # 替换空格和特殊字符为下划线
    cleaned_name = re.sub(r'\W+', '_', name)
    # 确保列名不以数字开头
    if cleaned_name[0].isdigit():
        cleaned_name = '_' + cleaned_name
    return cleaned_name
    
def factor_data_deal(element,train_time,sta_ids,time_freq,time_freq_data,time_freq_main,processing_methods):

    
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
    
    # 要素名
    # 组合字符串
    factor_name = []
    for i in range(len(elements)):
        combined_str = f"{elements[i]}_{time_freqs[i]}_{time_freq_data[i]}"
        cleaned_name = clean_column_name(combined_str)
        factor_name.append(cleaned_name)
    
    # result_dict = {}

    # for element, time_freq in zip(elements, time_freqs):
    #     if time_freq in result_dict:
    #         result_dict[time_freq] += ',' + element
    #     else:
    #         result_dict[time_freq] = element
            
    # elements = []
    # time_freqs = []
    # for key, value in result_dict.items():
    #     elements.append(value)
    #     time_freqs.append(key)

    # 4. 读取数据
    # 针对每个要素的时间尺度
    train_dataframes = []
    train_station_dataframes = []

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
        
        elif time_freqs[index] == 'Q':# ['%Y,%Y','3,4,5']
            train_time_use=[train_time,time_freq_data[index]]

        elif time_freqs[index] == 'M2':
            train_time_use=[train_time,time_freq_data[index]]
            
        elif time_freqs[index] == 'D1':
            train_time_use= train_time.split(',')[0]+'0101,'+train_time.split(',')[1]+'1231'
    
        train_data=data_read_sql(sta_ids,weather_element_str,train_time_use,weather_table_name,time_freqs[index])
        
        for name in ele.split(','):
            train_data.loc[train_data[name] > 1000, [name]] = np.nan

        
        #  5. 转换成年数据 或保留日数据
        if time_freq_main != 'D':
            if time_freqs[index] != 'D':
                train_station_data,train_data_deal=data_proce(train_data,processing_methods)
                train_station_data=train_station_data.rename(columns={ele:factor_name[index]})
                train_data_deal=train_data_deal.rename(columns={ele:factor_name[index]})

            else:
                train_station_data,train_data_deal=data_proce(train_data,processing_methods,additional_method='mean')
                train_station_data=train_station_data.rename(columns={ele:factor_name[index]})
                train_data_deal=train_data_deal.rename(columns={ele:factor_name[index]})
        else:
            train_station_data=train_data.copy()
            train_station_data.drop([ 'Station_Name'], axis=1,inplace=True)
            train_station_data=train_station_data.rename(columns={ele:factor_name[index]})

            train_data['Datetime'] = pd.to_datetime(train_data['Datetime'])
            train_data=train_data.set_index(train_data['Datetime'])
            train_data.drop([ 'Station_Id_C','Station_Name','Datetime'], axis=1,inplace=True)
            train_data_deal = train_data.resample('D').mean()
            train_data_deal=train_data_deal.rename(columns={ele:factor_name[index]})

        train_dataframes.append(train_data_deal)
        train_station_dataframes.append(train_station_data)

    combined_train = pd.concat(train_dataframes, axis=1)
    combined_train_station=train_station_dataframes[0].copy()
    for i in np.arange(len(train_station_dataframes))[1::]:
        combined_train_station= pd.merge(combined_train_station,train_station_dataframes[i], on=['Station_Id_C', '年'])

    return combined_train_station,combined_train
    
    
    

    
#%%
if __name__=='__main__':
    
    element='TEM_Avg,PRE_Time_2020,TEM_Avg'
    time_freq='Y,Q,Q'
    time_freq_data=['0','3,4,5','1']

    
    train_time='2020,2021'
    sta_ids='51886,52737,52876'
    time_freq_main='Y'
    
    resample_max = ['TEM_Max', 'PRS_Max', 'WIN_S_Max', 'WIN_S_Inst_Max', 'GST_Max', 'huangku']
    
    resample_min = ['TEM_Min', 'PRS_Min', 'GST_Min', 'RHU_Min', 'fanqing']
    
    resample_sum = ['SSH','PRE_Time_2020', 'PRE_Days', 'EVP_Big', 'EVP', 'EVP_Taka', 'PMET','sa','rainstorm','light_snow','snow',
                    'medium_snow','heavy_snow','severe_snow','Hail_Days','Hail','GaWIN',
                    'GaWIN_Days','SaSt','SaSt_Days','FlSa','FlSa_Days','FlDu','FlDu_Days',
                    'Thund','Thund_Days''high_tem','drought','light_drought','medium_drought',
                    'heavy_drought','severe_drought','Accum_Tem']
    
    resample_mean = ['TEM_Avg', 'PRS_Avg', 'WIN_S_2mi_Avg', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Avg_5cm', 'GST_Avg_10cm', 
                     'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 
                     'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 'SSH', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg', 'Cov', 'pmet']
    
    processing_methods = {element: 'mean' for element in resample_mean}
    processing_methods.update({element: 'sum' for element in resample_sum})
    processing_methods.update({element: 'max' for element in resample_max})
    processing_methods.update({element: 'min' for element in resample_min})
    
    
    train_station,train_data=factor_data_deal(element,train_time,sta_ids,time_freq,time_freq_data,time_freq_main,processing_methods)
        
        
        
        
        
        
        
        
        
        
        
        
        
   


