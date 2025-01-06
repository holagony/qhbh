# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:43:32 2024

@author: EDY
"""

import pandas as pd
import numpy as np
from Utils.data_processing import data_processing
from sklearn.linear_model import LinearRegression


def tem_table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year,l_data=None,n_data=None):

    #%% 数据前处理
    if ele == 'DTR':
        data_df['DTR']=data_df['TEM_Max']-data_df['TEM_Min']
        refer_df['DTR']=refer_df['TEM_Max']-refer_df['TEM_Min']
        nearly_df['DTR']=nearly_df['TEM_Max']-nearly_df['TEM_Min']
    
    #%% 要素匹配
    ele_ment=dict()
    ele_ment['TN10p']='TEM_Min'
    ele_ment['TX10p']='TEM_Max'
    ele_ment['TN90p']='TEM_Min'
    ele_ment['TX90p']='TEM_Max'
    ele_ment['ID']='TEM_Max'
    ele_ment['FD']='TEM_Min'
    ele_ment['TNx']='TEM_Min'
    ele_ment['TXx']='TEM_Max'
    ele_ment['TNn']='TEM_Min'
    ele_ment['TXn']='TEM_Max'
    ele_ment['DTR']='DTR'
    ele_ment['WSDI']='TEM_Max'
    ele_ment['CSDI']='TEM_Max'
    ele_ment['SU']='TEM_Max'
    ele_ment['TR']='TEM_Min'
    ele_ment['GSL']='TEM_Avg'
    ele_ment['high_tem']='TEM_Avg'

    
    last_df = nearly_df[nearly_df.index.year==last_year]
    last_df = last_df.pivot_table(index=last_df.index, columns=['Station_Id_C'], values=ele_ment[ele]) # 近1年df
    data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values=ele_ment[ele]) # 统计时段df
    refer_df = refer_df.pivot_table(index=refer_df.index, columns=['Station_Id_C'], values=ele_ment[ele]) # 参考时段df
    nearly_df = nearly_df.pivot_table(index=nearly_df.index, columns=['Station_Id_C'], values=ele_ment[ele]) # 近10年df
    data_df = data_df.round(1)

    #%% 要素计算
    # 冷夜日数 TN10p or 冷昼日数 TX10p
    if ele == 'TN10p' or ele == 'TX10p':
        for i in np.arange(np.size(data_df,1)):
            if i==0:
                l_data=l_data/100
            
            refer_sta=refer_df.iloc[:,i]
            refer_percentile_10 = refer_sta.quantile(l_data)    
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] < refer_percentile_10)).astype(int)
            
            last_sta=last_df.iloc[:,i]
            # last_percentile_10 = last_sta.quantile(l_data)    
            last_df.iloc[:,i] = ((last_df.iloc[:,i] < refer_percentile_10)).astype(int)
        
            data_sta=data_df.iloc[:,i]
            # data_percentile_10 = data_sta.quantile(l_data)    
            data_df.iloc[:,i] = ((data_df.iloc[:,i] < refer_percentile_10)).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            # nearly_percentile_10 = nearly_sta.quantile(l_data)    
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] < refer_percentile_10)).astype(int)
        
    # 暖夜日数 TN90p or 暖昼日数 TX90p
    elif ele == 'TN90p' or ele == 'TX90p':
        for i in np.arange(np.size(data_df,1)):
            if i==0:
                n_data=n_data/100

            
            refer_sta=refer_df.iloc[:,i]
            refer_percentile_90 = refer_sta.quantile(n_data)    
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] > refer_percentile_90)).astype(int)
            
            last_sta=last_df.iloc[:,i]
            # last_percentile_90 = last_sta.quantile(n_data)    
            last_df.iloc[:,i] = ((last_df.iloc[:,i] > refer_percentile_90)).astype(int)
        
            data_sta=data_df.iloc[:,i]
            # data_percentile_90 = data_sta.quantile(n_data)    
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > refer_percentile_90)).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            # nearly_percentile_90 = nearly_sta.quantile(n_data)    
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] > refer_percentile_90)).astype(int)
        
    # 结冰日数 ID or 霜冻日数 FD
    elif ele == 'ID' or ele == 'FD':
    
        for i in np.arange(np.size(data_df,1)):
            
            last_sta=last_df.iloc[:,i]
            last_df.iloc[:,i] = ((last_df.iloc[:,i] < 0)).astype(int)
        
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] < 0)).astype(int)
            
            refer_sta=refer_df.iloc[:,i]
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] < 0)).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] < 0)).astype(int)
    
    # 暖持续指数 WSDI:
    elif ele == 'WSDI':
        
         for i in np.arange(np.size(data_df,1)):
             
             refer_sta=refer_df.iloc[:,i]
             refer_percentile_10 = refer_sta.quantile(0.9) 
             refer_rolling = refer_sta.rolling(window=6)
             refer_rolling_min = refer_rolling.min()
             refer_df.iloc[:,i] = ((refer_rolling_min > refer_percentile_10)).astype(int)
             
             last_sta=last_df.iloc[:,i]
             # last_percentile_90 = last_sta.quantile(0.9) 
             last_rolling = last_sta.rolling(window=6)
             last_rolling_min =  last_rolling.min()
             last_df.iloc[:,i] = ((last_rolling_min > refer_percentile_10)).astype(int)
         
             data_sta=data_df.iloc[:,i]
             # data_percentile_10 = data_sta.quantile(0.9)  
             data_rolling = data_sta.rolling(window=6)
             data_rolling_min = data_rolling.min()
             data_df.iloc[:,i] = ((data_rolling_min > refer_percentile_10)).astype(int)
                          
             nearly_sta=nearly_df.iloc[:,i]
             # nearly_percentile_10 = nearly_sta.quantile(0.9)  
             nearly_rolling = nearly_sta.rolling(window=6)
             nearly_rolling_min = nearly_rolling.min()
             nearly_df.iloc[:,i] = ((nearly_rolling_min > refer_percentile_10)).astype(int)
             
    # 冷持续指数 CSDI:
    elif ele == 'CSDI':
        
         for i in np.arange(np.size(data_df,1)):
             
             refer_sta=refer_df.iloc[:,i]
             refer_percentile_10 = refer_sta.quantile(0.1) 
             refer_rolling = refer_sta.rolling(window=6)
             refer_rolling_min = refer_rolling.min()
             refer_df.iloc[:,i] = ((refer_rolling_min < refer_percentile_10)).astype(int)
             
             last_sta=last_df.iloc[:,i]
             # last_percentile_90 = last_sta.quantile(0.1) 
             last_rolling = last_sta.rolling(window=6)
             last_rolling_min =  last_rolling.min()
             last_df.iloc[:,i] = ((last_rolling_min < refer_percentile_10)).astype(int)
         
             data_sta=data_df.iloc[:,i]
             # data_percentile_10 = data_sta.quantile(0.1)  
             data_rolling = data_sta.rolling(window=6)
             data_rolling_min = data_rolling.min()
             data_df.iloc[:,i] = ((data_rolling_min < refer_percentile_10)).astype(int)
                          
             nearly_sta=nearly_df.iloc[:,i]
             # nearly_percentile_10 = nearly_sta.quantile(0.1)  
             nearly_rolling = nearly_sta.rolling(window=6)
             nearly_rolling_min = nearly_rolling.min()
             nearly_df.iloc[:,i] = ((nearly_rolling_min < refer_percentile_10)).astype(int)
    
    # 夏季日数 SU
    elif ele == 'SU':
    
        for i in np.arange(np.size(data_df,1)):
            
            last_sta=last_df.iloc[:,i]
            last_df.iloc[:,i] = ((last_df.iloc[:,i] > 25)).astype(int)
        
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > 25)).astype(int)
            
            refer_sta=refer_df.iloc[:,i]
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] > 25)).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] > 25)).astype(int)
    
    # 高温日数       
    elif ele == 'high_tem':
    
        for i in np.arange(np.size(data_df,1)):
            
            last_sta=last_df.iloc[:,i]
            last_df.iloc[:,i] = ((last_df.iloc[:,i] > n_data)).astype(int)
        
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > n_data)).astype(int)
            
            refer_sta=refer_df.iloc[:,i]
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] > n_data)).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] > n_data)).astype(int)
            
    # 热夜日数 TR
    elif ele == 'TR':
    
        for i in np.arange(np.size(data_df,1)):
            
            last_sta=last_df.iloc[:,i]
            last_df.iloc[:,i] = ((last_df.iloc[:,i] > 20)).astype(int)
        
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] > 20)).astype(int)
            
            refer_sta=refer_df.iloc[:,i]
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] > 20)).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] > 20)).astype(int)
    
    # 生长期长度 GSL:
    elif ele == 'GSL':
        
         for i in np.arange(np.size(data_df,1)):
             
             last_sta=last_df.iloc[:,i]
             last_rolling = last_sta.rolling(window=6)
             last_rolling_min =  last_rolling.min()
             last_df.iloc[:,i] = ((last_rolling_min  > 5)).astype(int)
         
             data_sta=data_df.iloc[:,i]
             data_rolling = data_sta.rolling(window=6)
             data_rolling_min = data_rolling.min()
             data_df.iloc[:,i] = ((data_rolling_min > 5)).astype(int)
             
             refer_sta=refer_df.iloc[:,i]
             refer_rolling = refer_sta.rolling(window=6)
             refer_rolling_min = refer_rolling.min()
             refer_df.iloc[:,i] = ((refer_rolling_min > 5)).astype(int)
             
             nearly_sta=nearly_df.iloc[:,i]
             nearly_rolling = nearly_sta.rolling(window=6)
             nearly_rolling_min = nearly_rolling.min()
             nearly_df.iloc[:,i] = ((nearly_rolling_min > 5)).astype(int)        
    #%% 数据转换
      
    if ele in ['TN10p', 'TX10p', 'TN90p', 'TX90p', 'ID', 'FD', 'SU','TR','GSL','high_tem']:
       
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').sum().astype(float).round(1)
        refer_df = refer_df.resample('Y').sum().astype(float).round(1)
        nearly_df = nearly_df.resample('Y').sum().astype(float).round(1)
        last_df = last_df.resample('Y').sum().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')
        refer_df.index = refer_df.index.strftime('%Y')
        nearly_df.index = nearly_df.index.strftime('%Y')
        last_df.index = last_df.index.strftime('%Y')
        
        # elif time_freq in ['M1','M2']:
        #     data_df = data_df.resample('M').sum()
        #     refer_df = refer_df.resample('M').sum()
        #     nearly_df = nearly_df.resample('M').sum()
        #     last_df = last_df.resample('M').sum()
            
        #     data_df.index = data_df.index.strftime('%Y-%m')
        #     refer_df.index = refer_df.index.strftime('%Y-%m')
        #     nearly_df.index = nearly_df.index.strftime('%Y-%m')
        #     last_df.index = last_df.index.strftime('%Y-%m')
    
    
    elif ele == 'TNx' or ele == 'TXx' :
        
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').max().astype(float).round(1)
        refer_df = refer_df.resample('Y').max().astype(float).round(1)
        nearly_df = nearly_df.resample('Y').max().astype(float).round(1)
        last_df = last_df.resample('Y').max().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')
        refer_df.index = refer_df.index.strftime('%Y')
        nearly_df.index = nearly_df.index.strftime('%Y')
        last_df.index = last_df.index.strftime('%Y')
        
        # elif time_freq in ['M1','M2']:
        #     data_df = data_df.resample('M').max()
        #     refer_df = refer_df.resample('M').max()
        #     nearly_df = nearly_df.resample('M').max()
        #     last_df = last_df.resample('M').max()
            
        #     data_df.index = data_df.index.strftime('%Y-%m')
        #     refer_df.index = refer_df.index.strftime('%Y-%m')
        #     nearly_df.index = nearly_df.index.strftime('%Y-%m')
        #     last_df.index = last_df.index.strftime('%Y-%m')
    
    
    elif ele == 'TNn' or ele == 'TXn' :
        
        # if time_freq in ['Y','Q']:
            
        data_df = data_df.resample('Y').min().astype(float).round(1)
        refer_df = refer_df.resample('Y').min().astype(float).round(1)
        nearly_df = nearly_df.resample('Y').min().astype(float).round(1)
        last_df = last_df.resample('Y').min().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')
        refer_df.index = refer_df.index.strftime('%Y')
        nearly_df.index = nearly_df.index.strftime('%Y')
        last_df.index = last_df.index.strftime('%Y')
        
    # elif ele == 'TNn' or ele == 'TXn' :
        
    #     # if time_freq in ['Y','Q']:
            
    #     data_df = data_df.resample('Y').min().astype(float).round(1)
    #     refer_df = refer_df.resample('Y').min().astype(float).round(1)
    #     nearly_df = nearly_df.resample('Y').min().astype(float).round(1)
    #     last_df = last_df.resample('Y').min().astype(float).round(1)
    
    #     data_df.index = data_df.index.strftime('%Y')
    #     refer_df.index = refer_df.index.strftime('%Y')
    #     nearly_df.index = nearly_df.index.strftime('%Y')
    #     last_df.index = last_df.index.strftime('%Y')
            
        # elif time_freq in ['M1','M2']:
        #     data_df = data_df.resample('M').min()
        #     refer_df = refer_df.resample('M').min()
        #     nearly_df = nearly_df.resample('M').min()
        #     last_df = last_df.resample('M').min()
            
        #     data_df.index = data_df.index.strftime('%Y-%m')
        #     refer_df.index = refer_df.index.strftime('%Y-%m')
        #     nearly_df.index = nearly_df.index.strftime('%Y-%m')
        #     last_df.index = last_df.index.strftime('%Y-%m')
        
    if ele in ['DTR','WSDI','CSDI']:
        
        data_df = data_df.resample('Y').mean().astype(float).round(1)
        refer_df = refer_df.resample('Y').mean().astype(float).round(1)
        nearly_df = nearly_df.resample('Y').mean().astype(float).round(1)
        last_df = last_df.resample('Y').mean().astype(float).round(1)
    
        data_df.index = data_df.index.strftime('%Y')
        refer_df.index = refer_df.index.strftime('%Y')
        nearly_df.index = nearly_df.index.strftime('%Y')
        last_df.index = last_df.index.strftime('%Y')
    
    def trend_rate(x):
        '''
        计算变率（气候倾向率）的pandas apply func
        '''
        try:
            x = x.to_frame()
            x['num'] = np.arange(len(x))
            x.dropna(how='any',inplace=True)
            train_x = x.iloc[:,-1].values.reshape(-1,1)
            train_y = x.iloc[:,0].values.reshape(-1,1)
            model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
            weight = model.coef_[0][0].round(3) * 10
            return weight
        except:
            return np.nan
        
    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df.iloc[:,:].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df.apply(trend_rate,axis=0).round(1)
    tmp_df.loc['最大值'] = data_df.iloc[:,:].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[:,:].min(axis=0)
    tmp_df.loc['与上一年比较值'] = (data_df.iloc[:,:].mean(axis=0) - last_df.iloc[:,:].mean(axis=0)).round(1)
    tmp_df.loc['近10年均值'] = nearly_df.iloc[:,:].mean(axis=0).round(1)
    tmp_df.loc['与近10年比较值'] = (data_df.iloc[:,:].mean(axis=0) - nearly_df.iloc[:,:].mean(axis=0)).round(1)
    tmp_df.loc['参考时段均值'] = refer_df.iloc[:,:].mean(axis=0).round(1)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']).round(1)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平']/tmp_df.loc['参考时段均值'])*100).round(2)
    
    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = stats_result.iloc[:,:].mean(axis=1).round(1)
    stats_result['区域距平'] = (stats_result.iloc[:,:].mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).round(1)
    stats_result['区域距平百分率'] = (stats_result['区域距平']/refer_df.iloc[:, :].mean().mean()).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:,:].max(axis=1)
    stats_result['区域最小值'] = stats_result.iloc[:,:].min(axis=1)

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
    stats_result = pd.concat((stats_result,tmp_df),axis=0)
    
    # index处理
    # if time_freq in ['Y','Q']:
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    # elif time_freq in ['M1','M2']:
    #     stats_result.insert(loc=0, column='时间', value=stats_result.index)
    # elif time_freq in ['D1','D2']:
    #     stats_result.insert(loc=0, column='时间', value=stats_result.index)
        
    stats_result.reset_index(drop=True, inplace=True)
    post_data_df = data_df.copy()
    post_refer_df = refer_df.copy()
    
    return stats_result, post_data_df, post_refer_df, reg_params


if __name__ == '__main__':
    
    #%% 数据读取
    path = r'D:\Project\3_项目\2_气候评估和气候可行性论证\qhkxxlz\Files\test_data\qh_day.csv'
    df = pd.read_csv(path, low_memory=False)
    df = data_processing(df)
    data_df = df[df.index.year<=2011]
    refer_df = df[(df.index.year>2000) & (df.index.year<2020)]
    nearly_df = df[df.index.year>2011]
    last_year = 2023
    time_freq = 'M1'
    ele='TN10p'

    stats_result, post_data_df, post_refer_df=tem_table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)