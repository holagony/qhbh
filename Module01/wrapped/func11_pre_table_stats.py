# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:37:05 2024

@author: EDY
"""

import pandas as pd
import numpy as np
from Utils.data_processing import data_processing
from sklearn.linear_model import LinearRegression

def persistent_time(df,time_freq):
    
    df['group'] = (df.iloc[:,0] != df.iloc[:,0].shift()).cumsum()
    
    if time_freq in ['Y','Q']:
        df['year'] = df.index.year            
        df['group'] = df['year'].astype(str) + '_' + df['group'].astype(str) 
    elif time_freq in ['M1','M2']:
        df['month'] = df.index.month 
        df['year'] = df.index.year            
        df['group'] = df['year'].astype(str) + '_' + df['month'].astype(str)+ '_' + df['group'].astype(str) 
           
    group_sums = df.groupby('group')[df.columns[0]].sum()
    last_ones = df[df[df.columns[0]] == 1].groupby('group').last().index
    group_sum_dict = group_sums.to_dict()
    df['result'] = 0
    for group_id in last_ones:
        last_one_index = df[df['group'] == group_id].index[-1]
        df.at[last_one_index, 'result'] = group_sum_dict[group_id]
        
    return df
                

def pre_table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year,R=None,R_flag=None,RD=None,RD_flag=None,Rxxday=None):

    last_df = nearly_df[nearly_df.index.year==last_year]
    last_df = last_df.pivot_table(index=last_df.index, columns=['Station_Id_C'], values='PRE_Time_2020') # 近1年df
    data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values='PRE_Time_2020') # 统计时段df
    refer_df = refer_df.pivot_table(index=refer_df.index, columns=['Station_Id_C'], values='PRE_Time_2020') # 参考时段df
    nearly_df = nearly_df.pivot_table(index=nearly_df.index, columns=['Station_Id_C'], values='PRE_Time_2020') # 近10年df
    data_df = data_df.round(1)

    D=dict()
    D['RZD']=0
    D['SDII']=1
    D['R25D']=25
    D['R10D']=10
    D['R50D']=50

    #%% 要素计算
    # 持续干期 CDD
    if ele == 'CDD':
        for i in np.arange(np.size(data_df,1)):
            
            last_sta=last_df.iloc[:,i].to_frame()
            last_sta.columns = last_sta.columns.get_level_values(0)
            last_sta_1= ((last_sta == 0)).astype(int)
            last_sta_2=persistent_time(last_sta_1,time_freq)
            last_df.iloc[:,i] =last_sta_2['result'].astype(float).round(1)
        
            data_sta=data_df.iloc[:,i].to_frame()
            data_sta.columns = data_sta.columns.get_level_values(0)
            data_sta_1= ((data_sta == 0)).astype(int)
            data_sta_2=persistent_time(data_sta_1,time_freq)
            data_df.iloc[:,i] =data_sta_2['result'].astype(float).round(1)

            refer_sta=refer_df.iloc[:,i].to_frame()
            refer_sta.columns = refer_sta.columns.get_level_values(0)
            refer_sta_1= ((refer_sta == 0)).astype(int)
            refer_sta_2=persistent_time(refer_sta_1,time_freq)
            refer_df.iloc[:,i] =refer_sta_2['result'].astype(float).round(1)
            
            nearly_sta=nearly_df.iloc[:,i].to_frame()
            nearly_sta.columns = nearly_sta.columns.get_level_values(0)
            nearly_sta_1= ((nearly_sta == 0)).astype(int)
            nearly_sta_2=persistent_time(nearly_sta_1,time_freq)
            nearly_df.iloc[:,i] =nearly_sta_2['result'].astype(float).round(1)
                        
    # 持续湿期 CWD
    if ele == 'CWD':
        for i in np.arange(np.size(data_df,1)):
            
            last_sta=last_df.iloc[:,i].to_frame()
            last_sta.columns = last_sta.columns.get_level_values(0)
            last_sta_1= ((last_sta > 0)).astype(int)
            last_sta_2=persistent_time(last_sta_1,time_freq)
            last_df.iloc[:,i] =last_sta_2['result'].astype(float).round(1)
        
            data_sta=data_df.iloc[:,i].to_frame()
            data_sta.columns = data_sta.columns.get_level_values(0)
            data_sta_1= ((data_sta > 0)).astype(int)
            data_sta_2=persistent_time(data_sta_1,time_freq)
            data_df.iloc[:,i] =data_sta_2['result'].astype(float).round(1)

            refer_sta=refer_df.iloc[:,i].to_frame()
            refer_sta.columns = refer_sta.columns.get_level_values(0)
            refer_sta_1= ((refer_sta > 0)).astype(int)
            refer_sta_2=persistent_time(refer_sta_1,time_freq)
            refer_df.iloc[:,i] =refer_sta_2['result'].astype(float).round(1)
            
            nearly_sta=nearly_df.iloc[:,i].to_frame()
            nearly_sta.columns = nearly_sta.columns.get_level_values(0)
            nearly_sta_1= ((nearly_sta > 0)).astype(int)
            nearly_sta_2=persistent_time(nearly_sta_1,time_freq)
            nearly_df.iloc[:,i] =nearly_sta_2['result'].astype(float).round(1)       

    # 降雨日数 降水强度 大雨日数 中雨日数 特强降水日数
    elif ele in ['RZD','SDII','R25D','R50D','R10D']:
    
        for i in np.arange(np.size(data_df,1)):
            
            last_sta=last_df.iloc[:,i]
            last_df.iloc[:,i] = ((last_df.iloc[:,i] >= D[ele])).astype(int)
        
            data_sta=data_df.iloc[:,i]
            data_df.iloc[:,i] = ((data_df.iloc[:,i] >= D[ele])).astype(int)
            
            refer_sta=refer_df.iloc[:,i]
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] >= D[ele])).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] >= D[ele])).astype(int)
            
    # 特强降水
    elif ele =='R50':
    
        for i in np.arange(np.size(data_df,1)):
            
            refer_sta=refer_df.iloc[:,i]
            refer_percentile_90 = refer_sta.quantile(0.95) 
            refer_df.iloc[((refer_df.iloc[:,i] < refer_percentile_90)),i] = 0
            
            last_sta=last_df.iloc[:,i]
            # last_percentile_90 = last_sta.quantile(0.95) 
            last_df.iloc[((last_df.iloc[:,i] < refer_percentile_90)),i] = 0
            
            data_sta=data_df.iloc[:,i]
            # data_percentile_90 = data_sta.quantile(0.95) 
            data_df.iloc[((data_df.iloc[:,i] < refer_percentile_90)),i] = 0
                        
            nearly_sta=nearly_df.iloc[:,i]
            # nearly_percentile_90 = nearly_sta.quantile(0.95) 
            nearly_df.iloc[((nearly_df.iloc[:,i] < refer_percentile_90)),i] = 0

    # 强降水
    elif ele =='R95%':
    
        for i in np.arange(np.size(data_df,1)):
            
            last_df.iloc[((last_df.iloc[:,i] < 50)),i] = 0
            data_df.iloc[((data_df.iloc[:,i] < 50)),i] = 0
            refer_df.iloc[((refer_df.iloc[:,i] < 50)),i] = 0
            nearly_df.iloc[((nearly_df.iloc[:,i] < 50)),i] = 0
            
    # 强降水日数 R95D
    elif ele == 'R95%D':
        for i in np.arange(np.size(data_df,1)):
            
                        
            refer_sta=refer_df.iloc[:,i]
            refer_percentile_90 = refer_sta.quantile(0.95)    
            refer_df.iloc[:,i] = ((refer_df.iloc[:,i] >= refer_percentile_90)).astype(int)
            
            last_sta=last_df.iloc[:,i]
            # last_percentile_90 = last_sta.quantile(0.95)    
            last_df.iloc[:,i] = ((last_df.iloc[:,i] >= refer_percentile_90)).astype(int)
        
            data_sta=data_df.iloc[:,i]
            # data_percentile_90 = data_sta.quantile(0.95)    
            data_df.iloc[:,i] = ((data_df.iloc[:,i] >= refer_percentile_90)).astype(int)
            
            nearly_sta=nearly_df.iloc[:,i]
            # nearly_percentile_90 = nearly_sta.quantile(0.95)    
            nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] >= refer_percentile_90)).astype(int)

    # 5日最大降水 Rx5day:
    elif ele == 'Rx5day':
        
         for i in np.arange(np.size(data_df,1)):
             
             last_sta=last_df.iloc[:,i]
             last_rolling = last_sta.rolling(window=5)
             last_rolling_sum =  last_rolling.sum()
             last_df.iloc[:,i] =last_rolling_sum.astype(float).round(1)
         
             data_sta=data_df.iloc[:,i]
             data_rolling = data_sta.rolling(window=5)
             data_rolling_sum = data_rolling.sum()
             data_df.iloc[:,i] = data_rolling_sum.astype(float).round(1)
             
             refer_sta=refer_df.iloc[:,i]
             refer_rolling = refer_sta.rolling(window=5)
             refer_rolling_sum = refer_rolling.sum()
             refer_df.iloc[:,i] = refer_rolling_sum.astype(float).round(1)
             
             nearly_sta=nearly_df.iloc[:,i]
             nearly_rolling = nearly_sta.rolling(window=5)
             nearly_rolling_sum = nearly_rolling.sum()
             nearly_df.iloc[:,i] = nearly_rolling_sum.astype(float).round(1)

    # 自定义降水
    elif ele =='R':
    
        for i in np.arange(np.size(data_df,1)):
            
            if R_flag==1:
                last_df.iloc[((last_df.iloc[:,i] <= R)),i] = 0
                data_df.iloc[((data_df.iloc[:,i] <= R)),i] = 0
                refer_df.iloc[((refer_df.iloc[:,i] <= R)),i] = 0
                nearly_df.iloc[((nearly_df.iloc[:,i] <= R)),i] = 0
            elif R_flag==2:
                last_df.iloc[((last_df.iloc[:,i] < R)),i] = 0
                data_df.iloc[((data_df.iloc[:,i] < R)),i] = 0
                refer_df.iloc[((refer_df.iloc[:,i] < R)),i] = 0
                nearly_df.iloc[((nearly_df.iloc[:,i] < R)),i] = 0
            elif R_flag==3:
                last_df.iloc[((last_df.iloc[:,i] > R)),i] = 0
                data_df.iloc[((data_df.iloc[:,i] > R)),i] = 0
                refer_df.iloc[((refer_df.iloc[:,i] > R)),i] = 0
                nearly_df.iloc[((nearly_df.iloc[:,i] > R)),i] = 0
            elif R_flag==4:
                last_df.iloc[((last_df.iloc[:,i] >= R)),i] = 0
                data_df.iloc[((data_df.iloc[:,i] >= R)),i] = 0
                refer_df.iloc[((refer_df.iloc[:,i] >= R)),i] = 0
                nearly_df.iloc[((nearly_df.iloc[:,i] >= R)),i] = 0
                
    # 自定义降水日
    elif ele =='RD':
    
        for i in np.arange(np.size(data_df,1)):
            
            if RD_flag==1:
                last_sta=last_df.iloc[:,i]
                last_df.iloc[:,i] = ((last_df.iloc[:,i] > RD)).astype(int)
            
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] > RD)).astype(int)
                
                refer_sta=refer_df.iloc[:,i]
                refer_df.iloc[:,i] = ((refer_df.iloc[:,i] > RD)).astype(int)
                
                nearly_sta=nearly_df.iloc[:,i]
                nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] > RD)).astype(int)
                
            elif RD_flag==2:
                last_sta=last_df.iloc[:,i]
                last_df.iloc[:,i] = ((last_df.iloc[:,i] >= RD)).astype(int)
            
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] >= RD)).astype(int)
                
                refer_sta=refer_df.iloc[:,i]
                refer_df.iloc[:,i] = ((refer_df.iloc[:,i] >= RD)).astype(int)
                
                nearly_sta=nearly_df.iloc[:,i]
                nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] >= RD)).astype(int)
                
            elif RD_flag==3:
                last_sta=last_df.iloc[:,i]
                last_df.iloc[:,i] = ((last_df.iloc[:,i] <= RD)).astype(int)
            
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] <= RD)).astype(int)
                
                refer_sta=refer_df.iloc[:,i]
                refer_df.iloc[:,i] = ((refer_df.iloc[:,i] <= RD)).astype(int)
                
                nearly_sta=nearly_df.iloc[:,i]
                nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] <= RD)).astype(int)
                
            elif RD_flag==4:
                last_sta=last_df.iloc[:,i]
                last_df.iloc[:,i] = ((last_df.iloc[:,i] < RD)).astype(int)
            
                data_sta=data_df.iloc[:,i]
                data_df.iloc[:,i] = ((data_df.iloc[:,i] < RD)).astype(int)
                
                refer_sta=refer_df.iloc[:,i]
                refer_df.iloc[:,i] = ((refer_df.iloc[:,i] < RD)).astype(int)
                
                nearly_sta=nearly_df.iloc[:,i]
                nearly_df.iloc[:,i] = ((nearly_df.iloc[:,i] < RD)).astype(int)
                                    
    # x日最大降水 Rxxday:
    elif ele == 'Rxxday':
        
         for i in np.arange(np.size(data_df,1)):
             
             last_sta=last_df.iloc[:,i]
             last_rolling = last_sta.rolling(window=Rxxday)
             last_rolling_sum =  last_rolling.sum()
             last_df.iloc[:,i] =last_rolling_sum.astype(float).round(1)
         
             data_sta=data_df.iloc[:,i]
             data_rolling = data_sta.rolling(window=Rxxday)
             data_rolling_sum = data_rolling.sum()
             data_df.iloc[:,i] = data_rolling_sum.astype(float).round(1)
             
             refer_sta=refer_df.iloc[:,i]
             refer_rolling = refer_sta.rolling(window=Rxxday)
             refer_rolling_sum = refer_rolling.sum()
             refer_df.iloc[:,i] = refer_rolling_sum.astype(float).round(1)
             
             nearly_sta=nearly_df.iloc[:,i]
             nearly_rolling = nearly_sta.rolling(window=Rxxday)
             nearly_rolling_sum = nearly_rolling.sum()
             nearly_df.iloc[:,i] = nearly_rolling_sum.astype(float).round(1)
    #%% 数据转换
      
    if ele in ['RZ','RZD','SDII','R25D','R50D','R10D','R95%D','R95%','R50','R','RD']:
       
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
    
    
    elif ele in ['CDD','CWD','Rx1day','Rx5day','Rxxday']:
        
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
    ele='RD'

    stats_result, post_data_df, post_refer_df=tem_table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year,RD=5,RD_flag=1)