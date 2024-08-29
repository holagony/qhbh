# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:54:21 2024

@author: EDY
"""
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

def trend_rate(x):
    '''
    计算变率（气候倾向率）的pandas apply func
    '''
    try:
        x = x.to_frame()
        x['num'] = np.arange(len(x))
        x.dropna(how='any', inplace=True)
        train_x = x.iloc[:, -1].values.reshape(-1, 1)
        train_y = x.iloc[:, 0].values.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
        weight = model.coef_[0][0].round(3) * 10
        return weight
    except:
        return np.nan
    
def choose_mod_path(inpath, data_cource,insti, var, time_scale, yr, expri_i,res=None):
    ## cmip数据路径选择
    """
    :param inpath: 根路径目录
    :param insti: 数据机构
    :param var: 要素缩写
    :param time_scale: 时间尺度
    :param yr: 年份
    :return: 数据所在路径、文件名
    """
    if yr < 2015:
        expri = 'historical'
    else:
        expri = expri_i
        
    if insti == 'CNRM-CM6-1':
        data_grid = '_r1i1p1f2_gr_'
        
    elif (insti == 'BCC-CSM2-MR') & (yr < 2015):
        data_grid = '_r3i1p1f1_gn_'

    else:
        data_grid = '_r1i1p1f1_gn_'

    if time_scale == 'daily':
        path1 = 'daily'
        filen = var + '_day_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    elif time_scale == 'monthly':
        path1 = 'monthly'
        filen = var + '_month_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    elif time_scale == 'yearly':
        path1 = 'yearly'
        filen = var + '_year_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'
    else:
        path1 = time_scale
        filen = var + '_' + time_scale + '_' + insti + '_' + expri + data_grid + str(yr) + '0101-' + str(yr) + '1231.nc'

    if data_cource=='original':
        path = os.path.join(inpath, data_cource,path1,insti ,expri,var,filen)
    else:
        path = os.path.join(inpath, data_cource,res,path1,insti ,expri,var,filen)

    return path

def time_choose(time_freq,stats_times):        


    if time_freq== 'Y':
        # Y
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
            
    elif time_freq== 'Q':

        # Q
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
       
    elif time_freq== 'M1':
   
        # M1
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        
        start_year = start_time[:4]
        end_year = end_time[:4]
        
    elif time_freq== 'M2':
    
        # M2
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
       
    elif time_freq== 'D1':
    
        # D1
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        
        start_year = start_time[:4]
        end_year = end_time[:4]
        
    elif time_freq== 'D2':
    
    # D2

        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        
    return start_year,end_year

def data_deal(result_days):
    result_days.set_index(result_days['年'],inplace=True)
    result_days.drop(['年'], axis=1, inplace=True) 
    
    tmp_days_df = pd.DataFrame(columns=result_days.columns)
    tmp_days_df.loc['平均'] = result_days.iloc[:, :].mean(axis=0).round(1)
    tmp_days_df.loc['变率'] = result_days.apply(trend_rate, axis=0)
    tmp_days_df.loc['最大值'] = result_days.iloc[:, :].max(axis=0)
    tmp_days_df.loc['最小值'] = result_days.iloc[:, :].min(axis=0)
    
    # 合并所有结果
    stats_days_result = result_days.copy()
    stats_days_result['区域均值'] = result_days.iloc[:, :].mean(axis=1).round(1)
    stats_days_result['区域最大值'] = result_days.iloc[:, :].max(axis=1)
    stats_days_result['区域最小值'] = result_days.iloc[:, :].min(axis=1)
    
    stats_days_result = pd.concat((stats_days_result, tmp_days_df), axis=0)
    stats_days_result.insert(loc=0, column='时间', value=stats_days_result.index)
    stats_days_result.reset_index(drop=True, inplace=True)

    return stats_days_result

def data_deal_num(result_days):

    result_days.set_index(result_days['年'],inplace=True)
    result_days.drop(['年'], axis=1, inplace=True) 
    
    tmp_days_df = pd.DataFrame(columns=result_days.columns)
    tmp_days_df.loc['平均'] = result_days.iloc[1:, :].mean(axis=0).round(1)
    tmp_days_df.loc['变率'] = result_days.iloc[1:, :].apply(trend_rate, axis=0)
    tmp_days_df.loc['最大值'] = result_days.iloc[1:, :].max(axis=0)
    tmp_days_df.loc['最小值'] = result_days.iloc[1:, :].min(axis=0)
    
    # 合并所有结果
    stats_days_result = result_days.copy()
    stats_days_result['区域均值1'] = result_days.iloc[1:, 0::2].mean(axis=1).round(1)
    stats_days_result['区域均值2'] = result_days.iloc[1:, 1::2].mean(axis=1).round(1)
    
    stats_days_result['区域最大值1'] = result_days.iloc[1:, 0::2].max(axis=1)
    stats_days_result['区域最大值2'] = result_days.iloc[1:, 1::2].max(axis=1)
    
    stats_days_result['区域最小值1'] = result_days.iloc[1:, 0::2].min(axis=1)
    stats_days_result['区域最小值2'] = result_days.iloc[1:, 1::2].min(axis=1)
    
    stats_days_result = pd.concat((stats_days_result, tmp_days_df), axis=0)
    stats_days_result.insert(loc=0, column='时间', value=stats_days_result.index)
    stats_days_result.reset_index(drop=True, inplace=True)
    
    stats_days_result.at[0,'区域均值1'] = '开始日期'
    stats_days_result.at[0,'区域均值2'] = '结束日期'
    stats_days_result.at[0,'区域最大值1'] = '开始日期'
    stats_days_result.at[0,'区域最大值2'] = '结束日期'
    stats_days_result.at[0,'区域最小值1'] = '开始日期'
    stats_days_result.at[0,'区域最小值2'] = '结束日期'
    return stats_days_result

def data_deal_2(data_df,refer_df):

    if '年' in data_df.columns:
        data_df.set_index(data_df['年'],inplace=True)
        data_df.drop(['年'], axis=1, inplace=True) 

    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df.apply(trend_rate, axis=0)
    tmp_df.loc['最大值'] = data_df.iloc[:, :].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[:, :].min(axis=0)
    tmp_df.loc['参考时段均值'] = refer_df.iloc[:, 1:].mean(axis=0).round(1)
    tmp_df.loc['距平'] = tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']
    tmp_df.loc['距平百分率%'] = ((tmp_df.loc['距平'] / tmp_df.loc['参考时段均值']) * 100).round(2)

    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = data_df.iloc[:, :].mean(axis=1).round(1)
    stats_result['区域距平'] = (data_df.iloc[:, :].mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).round(1)
    stats_result['区域距平百分率%'] = ((stats_result['区域距平']/refer_df.iloc[:, :].mean().mean())*100).round(2)
    stats_result['区域最大值'] = data_df.iloc[:, :].max(axis=1)
    stats_result['区域最小值'] = data_df.iloc[:, :].min(axis=1)

    stats_days_result = pd.concat((stats_result, tmp_df), axis=0)
    stats_days_result.insert(loc=0, column='时间', value=stats_days_result.index)
    stats_days_result.reset_index(drop=True, inplace=True)
    
    return stats_days_result

def data_deal_num_2(data_df,refer_df):

    if '年' in data_df.columns:

        data_df.set_index(data_df['年'],inplace=True)
        data_df.drop(['年'], axis=1, inplace=True) 

    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df.iloc[1:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df.iloc[1:, :].apply(trend_rate, axis=0)
    tmp_df.loc['最大值'] = data_df.iloc[1:, :].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[1:, :].min(axis=0)
    tmp_df.loc['参考时段均值'] = refer_df.iloc[1:, :].mean(axis=0).round(1)
    tmp_df.loc['距平'] = tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']
    tmp_df.loc['距平百分率%'] = ((tmp_df.loc['距平'] / tmp_df.loc['参考时段均值']) * 100).round(2)

    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值1'] = data_df.iloc[1:, 0::2].mean(axis=1).round(1)
    stats_result['区域均值2'] = data_df.iloc[1:, 1::2].mean(axis=1).round(1)

    stats_result['区域最大值1'] = data_df.iloc[1:, 0::2].max(axis=1)
    stats_result['区域最大值2'] = data_df.iloc[1:, 1::2].max(axis=1)

    stats_result['区域最小值1'] = data_df.iloc[1:, 0::2].min(axis=1)
    stats_result['区域最小值2'] = data_df.iloc[1:, 1::2].min(axis=1)

    stats_result['区域距平1'] = (data_df.iloc[1:, 0::2].mean(axis=1) - tmp_df.loc['参考时段均值'].iloc[0::2].mean()).round(1)
    stats_result['区域距平2'] = (data_df.iloc[1:, 1::2].mean(axis=1) - tmp_df.loc['参考时段均值'].iloc[1::2].mean()).round(1)

    stats_result['区域距平百分率%1'] = ((stats_result['区域距平1']/refer_df.iloc[1:, 0::2].mean().mean())*100).round(2)
    stats_result['区域距平百分率%2'] = ((stats_result['区域距平2']/refer_df.iloc[1:, 1::2].mean().mean())*100).round(2)


    stats_result = pd.concat((stats_result, tmp_df), axis=0)
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    stats_result.reset_index(drop=True, inplace=True)
    
    stats_result.at[0,'区域均值1'] = '开始日期'
    stats_result.at[0,'区域均值2'] = '结束日期'
    stats_result.at[0,'区域最大值1'] = '开始日期'
    stats_result.at[0,'区域最大值2'] = '结束日期'
    stats_result.at[0,'区域最小值1'] = '开始日期'
    stats_result.at[0,'区域最小值2'] = '结束日期'
    stats_result.at[0,'区域距平1'] = '开始日期'
    stats_result.at[0,'区域距平2'] = '结束日期'
    stats_result.at[0,'区域距平百分率%1'] = '开始日期'
    stats_result.at[0,'区域距平百分率%2'] = '结束日期'
    
    return stats_result

def calculate_average_hd(pre_data,ele):
    scene_hd_sum_count = {}
        
    for insti_a, scenes in pre_data.items():
        for scene_a, values in scenes.items():
            # break
            hd_value = values.get(ele, 0)
            
            if ele =='HDTIME_NUM':
                first_line=hd_value.iloc[0,:].to_frame()
                hd_value = hd_value.drop(hd_value.index[0])
            scene_hd_sum_count.setdefault(scene_a, [0, 0])
            scene_hd_sum_count[scene_a][0] += hd_value
            scene_hd_sum_count[scene_a][1] += 1

    # 计算平均值
    scene_hd_average = {scene: total / count for scene, (total, count) in scene_hd_sum_count.items()}
    
    if ele =='HDTIME_NUM':
        for insti_a, scenes in scene_hd_average.items():
            scene_hd_average[insti_a] = pd.concat([first_line.T, scenes]).reset_index(drop=True)

    return scene_hd_average

def percentile_std(scene,insti,df,ele):
    
    df_example=df[insti[0]][scene[0]][ele].copy()
    if '年' in df_example.columns:
        df_example.drop(['年'], axis=1, inplace=True) 
    data = np.zeros((len(insti), len(scene), df_example.shape[0], df_example.shape[1]))
    
    for i in np.arange(len(insti)):
        for j in np.arange(len(scene)):
            
            df_data=df[insti[i]][scene[j]][ele]
            if '年' in df_data.columns:
                df_data.drop(['年'], axis=1, inplace=True)
                
            data[i,j,:,:]=df_data.to_numpy()
            
    df_example=df[insti[0]][scene[0]][ele].copy()
    result=dict()
    for j in np.arange(len(scene)):
        result[scene[j]]=dict()
        
        df_example.iloc[:,1:]=np.percentile(data[:,j,:,:], 25, axis=0)
        result[scene[j]]['p25'] =df_example.copy()
        
        df_example.iloc[:,1:]=np.percentile(data[:,j,:,:], 75, axis=0)
        result[scene[j]]['p75'] = df_example.copy()
        
        df_example.iloc[:,1:]=np.std(np.array(data[:,j,:,:]).astype(float), axis=0)
        result[scene[j]]['std'] = df_example.copy()

            
if __name__ == '__main__':
    
    insti= 'BCC-CSM2-MR,CanESM5'
    insti = insti.split(',')
    scene=['ssp126','ssp245']
    ele='HD'
    df=pre_data
    data=percentile_std(scene,insti,pre_data,ele)
