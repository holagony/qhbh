# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:37:23 2024

@author: EDY


参数说明：

:param element: 
    总辐射量：TR 
    直接辐射占比： PDR;
    日照时数：SH；
    有效日照天数：ASD
:param refer_times: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'
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

import numpy as np
import pandas as pd
import uuid
import psycopg2
from psycopg2 import sql
from Utils.config import cfg
from Module02.page_energy.wrapped.func00_function import choose_mod_path
from Module02.page_energy.wrapped.func00_function import time_choose
from Module02.page_energy.wrapped.func00_function import data_deal
from Module02.page_energy.wrapped.func00_function import data_deal_num
from Module02.page_energy.wrapped.func00_function import data_deal_2
from Module02.page_energy.wrapped.func00_function import data_deal_num_2
from Module02.page_energy.wrapped.func00_function import calculate_average_hd
from Module02.page_energy.wrapped.func00_function import percentile_std
from Module02.page_energy.wrapped.func00_function import percentile_std_time

from Module02.page_energy.wrapped.func05_solar_power_his import energy_solar_his
# from Module02.page_energy.wrapped.func02_winter_heating_his import winter_heating_his

#%% main
def energy_solar_power(data_json):

    
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
    data_dir=r'D:\Project\qh\Evaluate_Energy\data'
    # data_dir='/zipdata'

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
    
    #elem_info=['采暖度日','采暖日','采暖起始日_日序']
    elem_dict=dict()
    elem_dict['TR']='总辐射量'
    elem_dict['PDR']=' 直接辐射占比'
    elem_dict['SH']='日照时数'
    elem_dict['ASD']='有效日照天数'

    #%% 统计计算模块
    if element == 'TR':
        elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,Year,Mon,Day,v14311'
    elif element == 'PDR':
        elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,Year,Mon,Day,v14311'
    elif element == 'SH':
        elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,ssh'
    elif element == 'ASD':
        elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,ssh'    
    
    if element in ['TR','PDR']:
        table_name='qh_climate_radi_hour'
    elif element in ['SH','ASD']:
        table_name='qh_climate_cmadaas_day'

    # 评估数据
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
    sta_ids1 = tuple(sta_ids.split(','))
    query = sql.SQL(f"""
                    SELECT {elements}
                    FROM public.{table_name}
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
    
    cur.close()
    conn.close()
    
    
    refer_result= energy_solar_his(element,refer_df)
        
    refer_result['年'] = refer_result['年'].astype(str)
    refer_result_z=data_deal(refer_result)
    
    refer_result_z=refer_result_z
    
    # 加一个 站点站名字典
    station_id=refer_df['Station_Id_C'].unique()
    
    matched_stations = pd.merge(pd.DataFrame({'Station_Id_C': station_id}),refer_df[['Station_Id_C', 'Station_Name']],on='Station_Id_C')
    matched_stations_unique = matched_stations.drop_duplicates()

    station_name = matched_stations_unique['Station_Name'].values
    station_id=matched_stations_unique['Station_Id_C'].values
    station_dict=pd.DataFrame(columns=['站名','站号'])
    station_dict['站名']=station_name
    station_dict['站号']=station_id
    
    '''
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
    
    HD=calculate_average_hd(pre_data,'HD')
    HDD18=calculate_average_hd(pre_data,'HDD18')
    HDTIME=calculate_average_hd(pre_data,'HDTIME_NUM')

#%% 基准期    
    base_p=pd.DataFrame(columns=refer_result_days_z.columns[1:-3:])
    base_p.loc[0,:]=refer_result_days_z[refer_result_days_z['时间'] == '平均'][1::-3].iloc[0, :]
    base_p.loc[1,:]=refer_result_hdd18_z[refer_result_hdd18_z['时间'] == '平均'][1::-3].iloc[0, :]
    base_p.loc[2,:]=refer_result_start_end_num_z[refer_result_start_end_num_z['时间'] == '平均'][1::-3].iloc[0, :]
    base_p.insert(0, '要素', ['采暖日','采暖度日','采暖起始日_日序'])
    
#%% 单模式距平和距平百分率

    pre_data_result=dict()
    for i in insti:
        pre_data_result[i]=dict()
        for j in scene:
            pre_data_result[i][j]=dict()
            for ele_a in elem_info:
                if ele_a in ['采暖度日','采暖日']:
                    pre_data_result[i][j][ele_a]=data_deal_2(pre_data[i][j][elem_dict[ele_a]],refer_result_days,2)
                else:
                    pre_data_result[i][j][ele_a]=data_deal_num_2(pre_data[i][j][elem_dict[ele_a]],refer_result_days,2)
    '''    

    result_df=dict()
    result_df['站点']=station_dict.to_dict(orient='records')
    result_df['表格']=dict()

    result_df['表格']['历史']=dict()
    result_df['表格']['历史']=refer_result_z.to_dict(orient='records')
    '''    

#%% 结果保存   
    result_df['表格']['预估']=dict()
    for insti_a in insti:
        result_df['表格']['预估'][insti_a]=dict()
        for scene_a in scene:
            result_df['表格']['预估'][insti_a][scene_a]=dict()
            result_df['表格']['预估'][insti_a][scene_a]['采暖日']=data_deal_2(pre_data[insti_a][scene_a]['HD'],refer_result_days,1)
            result_df['表格']['预估'][insti_a][scene_a]['采暖度日']=data_deal_2(pre_data[insti_a][scene_a]['HDD18'],refer_result_hdd18,1)
            result_df['表格']['预估'][insti_a][scene_a]['采暖起始日_日期']=pre_data[insti_a][scene_a]['HDTIME']
            result_df['表格']['预估'][insti_a][scene_a]['采暖起始日_日序']=data_deal_num_2(pre_data[insti_a][scene_a]['HDTIME_NUM'],result_start_end_num,1)
    
    result_df['表格']['预估']['集合']=dict()
    for scens in scene:
        result_df['表格']['预估']['集合'][scens]=dict()
        result_df['表格']['预估']['集合'][scens]['采暖日']=data_deal_2(HD[scens],refer_result_days,1)
        result_df['表格']['预估']['集合'][scens]['采暖度日'] =data_deal_2(HDD18[scens],refer_result_hdd18,1)
        result_df['表格']['预估']['集合'][scens]['采暖起始日_日序']=data_deal_num_2(HDTIME[scens],result_start_end_num,1)
        
    result_df['时序图']=dict()
    result_df['时序图']['集合_多模式' ]=dict()
    result_df['时序图']['集合_多模式' ]['采暖日']=percentile_std(scene,insti,pre_data,'HD',refer_result_days)
    result_df['时序图']['集合_多模式' ]['采暖度日']=percentile_std(scene,insti,pre_data,'HDD18',refer_result_hdd18)
    result_df['时序图']['集合_多模式' ]['采暖起始日_日序']=percentile_std_time(scene,insti,pre_data,result_start_end_num)
    
    result_df['时序图']['单模式' ]=pre_data_result.copy()
    result_df['时序图']['单模式' ]['基准期']=base_p.copy()

    result_df_dict=dict()
    result_df_dict['表格']=dict()

    result_df_dict['表格']['历史']=dict()
    result_df_dict['表格']['历史']['采暖日']=refer_result_days_z.to_dict(orient='records')
    result_df_dict['表格']['历史']['采暖度日']=refer_result_hdd18_z.to_dict(orient='records')
    result_df_dict['表格']['历史']['采暖起始日_日期']=refer_result_start_end.to_dict(orient='records')
    result_df_dict['表格']['历史']['采暖起始日_日序']=refer_result_start_end_num_z.to_dict(orient='records')
    
    result_df_dict['表格']['预估']=dict()
    for insti_a in insti:
        result_df_dict['表格']['预估'][insti_a]=dict()
        for scene_a in scene:
            result_df_dict['表格']['预估'][insti_a][scene_a]=dict()
            result_df_dict['表格']['预估'][insti_a][scene_a]['采暖日']=data_deal_2(pre_data[insti_a][scene_a]['HD'],refer_result_days,1).to_dict(orient='records')
            result_df_dict['表格']['预估'][insti_a][scene_a]['采暖度日']=data_deal_2(pre_data[insti_a][scene_a]['HDD18'],refer_result_hdd18,1).to_dict(orient='records')
            result_df_dict['表格']['预估'][insti_a][scene_a]['采暖起始日_日期']=pre_data[insti_a][scene_a]['HDTIME'].to_dict(orient='records')
            result_df_dict['表格']['预估'][insti_a][scene_a]['采暖起始日_日序']=data_deal_num_2(pre_data[insti_a][scene_a]['HDTIME_NUM'],result_start_end_num,1).to_dict(orient='records')

    result_df_dict['表格']['预估']['集合']=dict()
    for scens in scene:
        result_df_dict['表格']['预估']['集合'][scens]=dict()
        result_df_dict['表格']['预估']['集合'][scens]['采暖日']=data_deal_2(HD[scens],refer_result_days,1).to_dict(orient='records')
        result_df_dict['表格']['预估']['集合'][scens]['采暖度日']=data_deal_2(HDD18[scens],refer_result_hdd18,1).to_dict(orient='records')
        result_df_dict['表格']['预估']['集合'][scens]['采暖起始日_日序']=data_deal_num_2(HDTIME[scens],result_start_end_num,1).to_dict(orient='records')
    '''    
    return result_df

    
if __name__ == '__main__':
    
    # 总辐射量：TR 
    # 直接辐射占比： PDR;
    # 日照时数：SH；
    # 有效日照天数：ASD    
    data_json = dict()
    data_json['element'] ='PDR'
    data_json['refer_times'] = '2023,2024'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '1990,2018'
    data_json['sta_ids'] = '52863,52754,56029,52874'
    data_json['data_cource'] = 'original'
    data_json['insti'] = 'BCC-CSM2-MR,CanESM5'
    # data_json['res'] ='1'
    
    result_df=energy_solar_power(data_json)
    result=pd.DataFrame(result_df['表格']['历史'])
