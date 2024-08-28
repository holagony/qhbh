# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:06:27 2024

    参数说明：
    
    :param element: 采暖度日： HDD18;采暖日： HD;暖起始日：HTIME  没啥用，都会跑
    :param refer_years: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'
    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'
        季 - 'Q'
        月(连续) - 'M1'
        月(区间) - 'M2' 
        日(连续) - 'D1'
        日(区间) - 'D2
        
     :param stats_times: 对应原型的统计时段
         (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'
         (2)当time_freq选择季Q。下载连续的月数据，处理成季数据（多下两个月数据），提取历年相应的季节数据，传：['%Y,%Y','3,4,5']，其中'3,4,5'为季度对应的月份 
         (3)当time_freq选择月(连续)M1。下载连续的月数据，传参：'%Y%m,%Y%m'
         (4)当time_freq选择月(区间)M2。下载连续的月数据，随后提取历年相应的月数据，传参：['%Y,%Y','11,12,1,2'] 前者年份，'11,12,1,2'为连续月份区间
         (5)当time_freq选择日(连续)D1。下载连续的日数据，传参：'%Y%m%d,%Y%m%d'
         (6)当time_freq选择日(区间)D2。直接调天擎接口，下载历年区间时间段内的日数据，传：['%Y,%Y','%m%d,%m%d'] 前者年份，后者区间
     
     :param sta_ids: 传入的站点，多站，传：'52866,52713,52714'   
     :param data_cource: 预估数据源，传：'original'， 'Delta'， 'RDF' ，'RF' 
     :param insti: 模式选择，传：'BCC-CSM2-MR,CanESM5' 
     :param res 分辨率 ，数据源为original时可不传
        传参：
        1 - '1'
        5 - '5'
        10 - '10'
        25 - '25' 
        50 - '50'
        100 - '100'

"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import uuid
import psycopg2
from psycopg2 import sql
from Utils.config import cfg
from sklearn.linear_model import LinearRegression
from Module02.wrapped.func01_winter_heating_pre import winter_heating_pre
from Module02.wrapped.func02_winter_heating_his import winter_heating_his

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
#%%


def energy_winter_heating(data_json):

    
    #%% 参数信息
    element = data_json['element']
    refer_times = data_json['refer_times']
    time_freq = data_json['time_freq']
    stats_times = data_json['stats_times']
    sta_ids = data_json['sta_ids']
    data_cource = data_json['data_cource']
    insti = data_json['insti']
    res = data_json.get('res', '1')
    
    #%% 固定信息
    # data_dir=r'D:\Project\qh\Evaluate_Energy\data'
    data_dir='/zipdata'

    res_d=dict()
    res_d['1']='0.01deg'
    res_d['5']='0.06deg'
    res_d['10']='0.10deg'
    res_d['25']='0.25deg'
    res_d['50']='0.50deg'
    res_d['100']='1deg'
   
    # nc要素
    var='tas'
    
    # 情景选择
    # 'ssp126','ssp245','ssp585','1.5℃'，'2.0℃'
    scene=['ssp126','ssp245']
    
    # 时间频率
    time_scale='daily'
    
    #%% 统计计算模块
    
    # 评估数据
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
    elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,TEM_Avg'
    sta_ids1 = tuple(sta_ids.split(','))
    query = sql.SQL(f"""
                    SELECT {elements}
                    FROM public.qh_qhbh_cmadaas_day
                    WHERE
                        CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                        AND station_id_c IN %s
                    """)
    
    start_year = refer_times.split(',')[0]
    end_year = refer_times.split(',')[1]
    
    cur.execute(query, (start_year, end_year,sta_ids1))
    data = cur.fetchall()
    refer_df = pd.DataFrame(data)
    refer_df.columns = elements.split(',')
    
    # 关闭数据库
    cur.close()
    conn.close()
    
    refer_df.set_index('Datetime', inplace=True)
    refer_df.index = pd.DatetimeIndex(refer_df.index)
    refer_df['Station_Id_C'] = refer_df['Station_Id_C'].astype(str)
    
    if 'Unnamed: 0' in refer_df.columns:
        refer_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    refer_result_days,refer_result_hdd18,refer_result_start_end,refer_result_start_end_num= winter_heating_his(refer_df)
        
    refer_result_days_z=data_deal(refer_result_days)
    refer_result_hdd18_z=data_deal(refer_result_hdd18)
    refer_result_start_end_num_z=data_deal_num(refer_result_start_end_num)
    
    # 预估数据
    insti = insti.split(',')
    sta_ids2=sta_ids.split(',')
    stats_start_year,stats_end_year=time_choose(time_freq,stats_times)
    
    pre_data=dict()
    for insti_a in insti:
        pre_data[insti_a]=dict()
        for scene_a in scene:
            pre_data[insti_a][scene_a]=dict()
            stats_path=[choose_mod_path(data_dir, data_cource,insti_a, var, time_scale, year_a, scene_a,res_d[res]) for year_a in np.arange(int(stats_start_year),int(stats_end_year)+1,1)]
            result_days,result_hdd18,result_start_end,result_start_end_num= winter_heating_pre(stats_path,sta_ids2,time_freq,stats_times)
    
            pre_data[insti_a][scene_a]['HDD18']=result_hdd18
            pre_data[insti_a][scene_a]['HD']=result_days
            pre_data[insti_a][scene_a]['HDTIME']=result_start_end
            pre_data[insti_a][scene_a]['HDTIME_NUM']=result_start_end_num
    
    #%% 求集合
    total_hdd18 = 0
    total_hd = 0
    total_htime_num = 0
    
    count = 0
    
    # 遍历pre_data字典，累加所有的HDD18值
    for insti_a in pre_data:
        for scene_a in pre_data[insti_a]:
            total_hdd18 += pre_data[insti_a][scene_a]['HDD18']
            total_hd += pre_data[insti_a][scene_a]['HD']
            total_htime_num += pre_data[insti_a][scene_a]['HDTIME_NUM'].iloc[1::,:]
    
            count += 1  # 每找到一个HDD18值，计数器加1
    
    # 计算HDD18的平均值
    average_hdd18 = total_hdd18 / count if count > 0 else 0
    average_hd = total_hd / count if count > 0 else 0
    average_htime_num = total_htime_num / count if count > 0 else 0
    first_row_df = result_start_end_num.iloc[0,:].to_frame().T  
    average_htime_num = pd.concat([first_row_df, average_htime_num], ignore_index=True)
    
    average_hdd18_z=data_deal_2(average_hdd18,refer_result_hdd18)
    average_hd_z=data_deal_2(average_hd,refer_result_days)
    average_htime_num_z=data_deal_num_2(average_htime_num,result_start_end_num)
    
    #%% 结果保存
    result_df=dict()
    result_df['历史']=dict()
    result_df['历史']['采暖日']=refer_result_days_z
    result_df['历史']['采暖度日']=refer_result_hdd18_z
    result_df['历史']['采暖起始日_日期']=refer_result_start_end
    result_df['历史']['采暖起始日_日序']=refer_result_start_end_num_z
    
    result_df['预估']=dict()
    for insti_a in insti:
        result_df['预估'][insti_a]=dict()
        for scene_a in scene:
            result_df['预估'][insti_a][scene_a]=dict()
            result_df['预估'][insti_a][scene_a]['采暖日']=data_deal_2(pre_data[insti_a][scene_a]['HD'],refer_result_hdd18)
            result_df['预估'][insti_a][scene_a]['采暖度日']=data_deal_2(pre_data[insti_a][scene_a]['HDD18'],refer_result_hdd18)
            result_df['预估'][insti_a][scene_a]['采暖起始日_日期']=pre_data[insti_a][scene_a]['HDTIME']
            result_df['预估'][insti_a][scene_a]['采暖起始日_日序']=data_deal_num_2(pre_data[insti_a][scene_a]['HDTIME_NUM'],result_start_end_num)
    
    result_df['预估']['集合']=dict()
    result_df['预估']['集合']['采暖日'] = average_hd_z
    result_df['预估']['集合']['采暖度日'] =average_hdd18_z
    result_df['预估']['集合']['采暖起始日_日序'] =average_htime_num_z

    
    result_df_dict=dict()
    result_df_dict['历史']=dict()
    result_df_dict['历史']['采暖日']=refer_result_days_z.to_dict()
    result_df_dict['历史']['采暖度日']=refer_result_hdd18_z.to_dict()
    result_df_dict['历史']['采暖起始日_日期']=refer_result_start_end.to_dict()
    result_df_dict['历史']['采暖起始日_日序']=refer_result_start_end_num_z.to_dict()
    
    result_df_dict['预估']=dict()
    for insti_a in insti:
        result_df_dict['预估'][insti_a]=dict()
        for scene_a in scene:
            result_df_dict['预估'][insti_a][scene_a]=dict()
            result_df_dict['预估'][insti_a][scene_a]['采暖日']=data_deal_2(pre_data[insti_a][scene_a]['HD'],refer_result_hdd18).to_dict()
            result_df_dict['预估'][insti_a][scene_a]['采暖度日']=data_deal_2(pre_data[insti_a][scene_a]['HDD18'],refer_result_hdd18).to_dict()
            result_df_dict['预估'][insti_a][scene_a]['采暖起始日_日期']=pre_data[insti_a][scene_a]['HDTIME'].to_dict()
            result_df_dict['预估'][insti_a][scene_a]['采暖起始日_日序']=data_deal_num_2(pre_data[insti_a][scene_a]['HDTIME_NUM'],result_start_end_num).to_dict()

    result_df_dict['预估']['集合']=dict()
    result_df_dict['预估']['集合']['采暖日'] = average_hd_z.to_dict()
    result_df_dict['预估']['集合']['采暖度日'] =average_hdd18_z.to_dict()
    result_df_dict['预估']['集合']['采暖起始日_日序'] =average_htime_num_z.to_dict()
    
    return result_df,result_df_dict
        
if __name__ == '__main__':
    
    data_json = dict()
    data_json['element'] ='HDTIME'
    data_json['refer_times'] = '2000,2010'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2015,2018'
    data_json['sta_ids'] = '52754,56151,52855,52862,56065,52645,56046,52955,52968,52963,52825,56067,52713,52943,52877,52633,52866'
    data_json['data_cource'] = 'original'
    data_json['insti'] = 'BCC-CSM2-MR,CanESM5'
    data_json['res'] ='1'
    
    result_df,result_df_dict=energy_winter_heating(data_json)
    
