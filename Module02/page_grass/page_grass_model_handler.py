# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:46:49 2024

@author: EDY

:param main_element：对应原型，评估要素
        草地返青期 grassland_green_period
        草地枯黄期 grassland_yellow_period
        草地生育期 grassland_growth_period
        草地覆盖度 grassland_coverage
        草地产量 grassland_yield
        植被生态指数 vegetation_index
        植被净初级生产力 vegetation_pri_productivity

    
:param sta_ids: 传入的站点，多站，传：'52866,52713,52714'

    
:param factor_element：对应原型，因子要素
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
           
:param factor_time_freq: 对应原型选择数据的时间尺度
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
        


"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Module02.page_ice.wrapped.func01_factor_data_deal import factor_data_deal
from Module02.page_grass.wrapped.func02_grass_evaluate_data_deal import grass_evaluate_data_deal


def grass_model_def(data_json):

    main_element= data_json['main_element']
    sta_ids= data_json['sta_ids']
    time_freq_main= data_json['time_freq_main']
    time_freq_main_data= data_json['time_freq_main_data']
    factor_element= data_json['factor_element']
    factor_time_freq= data_json['factor_time_freq']
    factor_time_freq_data= data_json['factor_time_freq_data']
    train_time= data_json['train_time']
    verify_time= data_json['verify_time']

    
    #%% step1 模型构建部分，主要用到站点数据作为训练和验证数据
    
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
        
    # 因子数据_训练_验证
    train_station,train_data=factor_data_deal(factor_element,train_time,sta_ids,factor_time_freq,factor_time_freq_data,time_freq_main,processing_methods)
    verify_station,verify_data=factor_data_deal(factor_element,verify_time,sta_ids,factor_time_freq,factor_time_freq_data,time_freq_main,processing_methods)
        
    train_data=train_data.reset_index()
    verify_data=verify_data.reset_index()
    train_data = train_data.rename(columns={'年份': 'Datetime'})
    verify_data = verify_data.rename(columns={'年份': 'Datetime'})
    train_data = train_data.set_index(train_data['Datetime'].astype(str))
    verify_data = verify_data.set_index(verify_data['Datetime'].astype(str))
    
    # 评估数据_训练_验证
    train_evaluate,train_evaluate_station=grass_evaluate_data_deal(main_element,train_time,sta_ids,time_freq_main,time_freq_main_data)
    verify_evaluate,verify_evaluate_station=grass_evaluate_data_deal(main_element,verify_time,sta_ids,time_freq_main,time_freq_main_data)
        
    # 拼接一下数据
    train=pd.merge(train_data,train_evaluate, left_index=True, right_index=True, how='inner')
    train.dropna(inplace=True)
    evaluate=pd.merge(verify_data,verify_evaluate, left_index=True, right_index=True, how='inner')
    evaluate.dropna(inplace=True)
    
    independent_columns =train_data.columns[1::]
    y_columns=train_evaluate.columns
    X = train[independent_columns].values
    y = train[y_columns].values
    
    # 模型构建
    model = LinearRegression()
    model.fit(X, y)
    
    # 验证因子数据
    X = evaluate[independent_columns].values
    y_pred = model.predict(X)
    
    # 相关因子
    intercept=np.round(model.intercept_,2) # 截距
    coef=np.round(model.coef_,2) # 系数
    R= np.round(np.corrcoef(evaluate.iloc[:,-1].astype(float), np.squeeze(y_pred))[0, 1],2)  # 相关系数
    AE=np.round(np.sum(abs(evaluate.iloc[:,-1].astype(float)-np.squeeze(y_pred))),2)  # 绝对误差
    RMSE=np.round(mean_squared_error(evaluate.iloc[:,-1].astype(float),np.squeeze(y_pred)),2)  # 均方根误差
    
    result=dict()
    result['系数']=dict()
    result['系数']['截距']=intercept[0]
    for index,i in enumerate(independent_columns):
        result['系数'][i]=coef[0,index]
    
    result['相关系数']=R
    result['绝对误差']=AE
    result['均方根误差']=RMSE

    return result
    

if __name__=='__main__':
    
    data_json = dict()
    data_json['main_element'] ='dwei'
    data_json['sta_ids'] = '52943,56021,56045,56065'
    data_json['time_freq_main'] = 'Y'
    data_json['time_freq_main_data'] = '0'
    data_json['factor_element'] ='TEM_Avg,PRE_Time_2020'
    data_json['factor_time_freq'] = 'Y,Q'
    data_json['factor_time_freq_data'] =['0','3,4,5']
    data_json['train_time'] = '2016,2023'
    data_json['verify_time'] = '2015,2023'


    result=grass_model_def(data_json)






