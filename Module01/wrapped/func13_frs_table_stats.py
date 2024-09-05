# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:32:24 2024

@author: EDY

查询统计 - 冰冻圈

最大冻结深度 FRS_DEPTH
开始冻结日期 FRS_START
完全融化日期 FRS_END
冻结期 FRS_TIME
"""
import pandas as pd
import numpy as np
from datetime import  date,datetime, timedelta
from sklearn.linear_model import LinearRegression


def frs_processing(element,df):
    
    if element in ['FRS_DEPTH']:
    # 最大冻结深度
        ele='frs_depth' 
        df = df.pivot_table(index=df.index, columns=['Station_Id_C'], values=ele)  # 参考时段df
        df.replace(999999, np.nan, inplace=True)
    
        # df = df.resample('Y').max()
        df = df.resample(rule='AS-SEP').max()

        df.index = df.index.strftime('%Y')
        result_df=df.copy()

        return result_df
    
    if element in ['FRS_START','FRS_END','FRS_TIME']:

        df["时间分组"] = df.index.year - (df.index.month < 9)
        grouped = df.groupby(["Station_Id_C", "时间分组"])
        start_times = grouped.apply(lambda x: x[x["frs_state"] == 2].index.min())
        end_times = grouped.apply(lambda x: x[x["frs_state"] == 2].index.max())
    
        start_times = start_times.reset_index()
        end_times = end_times.reset_index()
    
        start_times.columns = ["站名", "年", "开始时间"]
        end_times.columns = ["站名", "年", "结束时间"]
    
        start_df = start_times.pivot(index="年", columns="站名", values="开始时间")
        end_df = end_times.pivot(index="年", columns="站名", values="结束时间")
        if element in ['FRS_START']:
            result_start=start_df.copy()
            
            result_df=result_start.copy()
            for i in np.arange(np.size(result_start,0)):
                for j in np.arange(np.size(result_start,1)):
                    if pd.notna(result_start.iloc[i,j]):
                        result_df.iloc[i,j]=(result_start.iloc[i,j]-datetime(result_start.iloc[i,j].year, 1, 1)).days+1
                    else:
                        result_df.iloc[i,j]=999999
            result_df[result_df==999999]=np.nan    
            return result_df,result_start

        if element in ['FRS_END']:
            result_end=end_df.copy()

            result_df=result_end.copy()
            for i in np.arange(np.size(result_end,0)):
                for j in np.arange(np.size(result_end,1)):
                    if pd.notna(result_end.iloc[i,j]):
                        result_df.iloc[i,j]=(result_end.iloc[i,j]-datetime(result_end.iloc[i,j].year, 1, 1)).days+1
                    else:
                        result_df.iloc[i,j]=999999
            result_df[result_df==999999]=np.nan            
            return result_df,result_end
        
        if element in ['FRS_TIME']:
            data_len_df=end_df.copy()
            for i in np.arange(np.size(start_df,0)):
                for j in np.arange(np.size(start_df,1)):
                    if pd.notna(start_df.iloc[i,j]) & pd.notna(end_df.iloc[i,j]):
                        data_len_df.iloc[i,j]=(end_df.iloc[i,j]-start_df.iloc[i,j]).days
                    else:
                        data_len_df.iloc[i,j]=999999
            data_len_df[data_len_df==999999]=np.nan    
            result_df=data_len_df.copy()
            
            return result_df
            
def frs_table_stats(data_df, refer_df, nearly_df, ele, last_year):
    '''
    data_df 天擎统计时段数据
    refer_df 天擎参考时段数据
    nearly_df 天擎近10年数据
    time_freq 数据的时间类型 年/月/季/小时
    ele 计算的要素
    last_year 近1年年份
    '''
    last_df = nearly_df[nearly_df.index.year==last_year]


    if ele in ['FRS_TIME','FRS_DEPTH']:
        data_df=frs_processing(ele,data_df)
        refer_df=frs_processing(ele,refer_df)
        nearly_df=frs_processing(ele,nearly_df)
        last_df=frs_processing(ele,last_df)
        
    elif ele in ['FRS_START','FRS_END']:
        
        data_df,data_df_time=frs_processing(ele,data_df)
        refer_df,refer_df_time=frs_processing(ele,refer_df)
        nearly_df,nearly_df_time=frs_processing(ele,nearly_df)
        last_df,last_df_time=frs_processing(ele,last_df)

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
    tmp_df.loc['平均'] = pd.to_numeric(data_df.iloc[:,:].mean(axis=0),errors='coerce').astype(float).round(1)
    tmp_df.loc['变率'] = data_df.apply(trend_rate,axis=0).astype(float).round(1)
    tmp_df.loc['最大值'] = data_df.iloc[:,:].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[:,:].min(axis=0)
    tmp_df.loc['与上一年比较值'] = (data_df.iloc[:,:].mean(axis=0) - last_df.iloc[:,:].mean(axis=0)).round(1)
    tmp_df.loc['近10年均值'] = pd.to_numeric(nearly_df.iloc[:,:].mean(axis=0),errors='coerce').astype(float).round(1)
    tmp_df.loc['与近10年比较值'] = pd.to_numeric((data_df.iloc[:,:].mean(axis=0) - nearly_df.iloc[:,:].mean(axis=0)),errors='coerce').astype(float).round(1)
    tmp_df.loc['参考时段均值'] = refer_df.iloc[:,:].mean(axis=0).round(1)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']).astype(float).round(1)
    tmp_df.loc['距平百分率%'] = ((tmp_df.loc['距平']/tmp_df.loc['参考时段均值'])*100).astype(float).round(2)

    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = stats_result.iloc[:,:].mean(axis=1).astype(float).round(1)
    stats_result['区域距平'] = (stats_result.iloc[:,:].mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).astype(float).round(1)
    stats_result['区域距平百分率%'] = (stats_result['区域距平']/refer_df.iloc[:, :].mean().mean()).astype(float).round(1)
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

    stats_result.insert(loc=0, column='时间', value=stats_result.index)

    stats_result.reset_index(drop=True, inplace=True)
    post_data_df = data_df.copy()
    post_refer_df = refer_df.copy()
    
    if ele in ['FRS_TIME','FRS_DEPTH']:
        return stats_result, post_data_df, post_refer_df, reg_params
            
    elif ele in ['FRS_START','FRS_END']:
        
        return stats_result, post_data_df, post_refer_df, reg_params,data_df_time

        
            
            
            
            
            
            

    