# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:16:30 2024

@author: EDY

参数说明：

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
    10 - '10'
    25 - '25' 
    50 - '50'
    100 - '100'

"""
import os
import numpy as np
import pandas as pd
import uuid
from Utils.config import cfg
from Module02.page_energy.wrapped.func00_function import time_choose
from Module02.page_energy.wrapped.func00_function import data_deal_2

from Module02.page_extreme.wrapped.func03_cmip_data_deal import extreme_pre
from Module02.page_climate.wrapped.func_plot import interp_and_mask, plot_and_save


    
#%% main
def extreme_climate_esti(data_json):

    
    #%% 参数信息
    element = data_json['element']
    refer_times = data_json['refer_times']
    time_freq = data_json['time_freq']
    stats_times = data_json['evaluate_times']
    sta_ids = data_json['sta_ids']
    data_cource = data_json['cmip_type']
    insti = data_json['cmip_model']
    res = data_json.get('cmip_res', '25')
    plot= data_json.get('plot')
    shp_path=data_json['shp_path']
    method='idw'
    
    l_data = data_json.get('l_data')
    n_data = data_json.get('n_data')
    R = data_json.get('R')
    R_flag = data_json.get('R_flag')
    RD = data_json.get('RD')
    RD_flag = data_json.get('RD_flag')
    GaWIN = data_json.get('GaWIN')
    GaWIN_flag = data_json.get('GaWIN_flag')
    Rxxday = data_json.get('Rxxday')
    degree = data_json.get('degree')
    #%% 固定信息
    
    if shp_path is not None:
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径

    if os.name == 'nt':
        data_dir=r'D:\Project\qh'
    elif os.name == 'posix':
        data_dir='/model_data/station_data/csv'
    else:
        data_dir='/model_data/station_data/csv'

    if data_cource == 'original':
        res='25'
        
    res_d=dict()
    res_d['25']='0.25deg'
    res_d['50']='0.50deg'
    res_d['100']='1.00deg'
   
    if data_cource != 'original':
        data_dir=os.path.join('/model_data/station_data_delta/csv',res_d[res])
    
    # 情景选择
    # 'ssp126','ssp245','ssp585','1.5℃'，'2.0℃'
    scene=['ssp126','ssp245','ssp585']
    
    # 时间频率
    time_scale='daily'
    uuid4 = uuid.uuid4().hex
    data_out = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_out):
        os.makedirs(data_out)
        os.chmod(data_out, 0o007 | 0o070 | 0o700)
        
    #%% 要素字典
    ele_dict=dict()

    # 极端气温指数
    ele_dict['TN10p']='TEM_Min'
    ele_dict['TX10p']='TEM_Max'
    ele_dict['TN90p']='TEM_Min'
    ele_dict['TX90p']='TEM_Max'
    ele_dict['ID']='TEM_Max'
    ele_dict['FD']='TEM_Min'
    ele_dict['TNx']='TEM_Min'
    ele_dict['TXx']='TEM_Max'
    ele_dict['TNn']='TEM_Min'
    ele_dict['TXn']='TEM_Max'
    ele_dict['DTR']='TEM_Min,TEM_Max'
    ele_dict['WSDI']='TEM_Max'
    ele_dict['CSDI']='TEM_Max'
    ele_dict['SU']='TEM_Max'
    ele_dict['TR']='TEM_Min'
    ele_dict['GSL']='TEM_Avg'
    ele_dict['high_tem']='TEM_Max'

    # 极端降水指数
    ele_dict['CDD']='PRE_Time_2020'
    ele_dict['CWD']='PRE_Time_2020'
    ele_dict['RZ']='PRE_Time_2020'
    ele_dict['RZD']='PRE_Time_2020'
    ele_dict['SDII']='PRE_Time_2020'
    ele_dict['R95%']='PRE_Time_2020'
    ele_dict['R95%D']='PRE_Time_2020'
    ele_dict['R50']='PRE_Time_2020'
    ele_dict['R50D']='PRE_Time_2020'
    ele_dict['R10D']='PRE_Time_2020'
    ele_dict['R25D']='PRE_Time_2020'
    ele_dict['Rx1day']='PRE_Time_2020'
    ele_dict['Rx5day']='PRE_Time_2020'
    ele_dict['R']='PRE_Time_2020'
    ele_dict['RD']='PRE_Time_2020' 
    ele_dict['Rxxday']='PRE_Time_2020'
    
    # 其他气候指数
    ele_dict['GaWIN']='win_s_2mi_avg'
    ele_dict['drought']='drought'
    ele_dict['light_drought']='light_drought'
    ele_dict['medium_drought']='medium_drought'
    ele_dict['heavy_drought']='heavy_drought'
    ele_dict['severe_drought']='severe_drought'
    
    # 模式数据对应的变量
    nc_dict=dict()
    nc_dict['TEM_Avg']='tas'
    nc_dict['TEM_Max']='tasmax'
    nc_dict['TEM_Min']='tasmin'
    nc_dict['PRE_Time_2020']='pr'
    nc_dict['win_s_2mi_avg']='ws'
    nc_dict['TEM_Min,TEM_Max']='ws'

    nc_dict['drought']='drought'
    nc_dict['light_drought']='light_drought'
    nc_dict['medium_drought']='medium_drought'
    nc_dict['heavy_drought']='heavy_drought'
    nc_dict['severe_drought']='severe_drought'
    
    
    #%% 站点站名字典    
    df_station=pd.read_csv(cfg.FILES.STATION,encoding='gbk')
    df_station['区站号']=df_station['区站号'].astype(str)
    
    station_id=sta_ids.split(',')
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '站点名','纬度','经度']],on='区站号')
    matched_stations_unique = matched_stations.drop_duplicates()

    station_name = matched_stations_unique['站点名'].values
    station_id=matched_stations_unique['区站号'].values
    station_dict=pd.DataFrame(columns=['站名','站号'])
    station_dict['站名']=np.array([s[:s.find('国')] if '国' in s else s for s in station_name])
    station_dict['站号']=station_id
    lon_list=matched_stations_unique['经度'].values
    lat_list=matched_stations_unique['纬度'].values 
    
    #%% 参考时段 基准期
    sta_ids2=sta_ids.split(',')
    
    refer_data=dict()
    base_p=dict()
    for scene_a in scene:
        refer_data[scene_a]=dict()
        base_p[scene_a]=dict()
        for insti_a in insti:
            refer_data[scene_a][insti_a]=dict()
            base_p[scene_a][insti_a]=dict()

            cmip_result=extreme_pre(element,data_dir,time_scale,insti_a,scene_a,nc_dict[ele_dict[element]],refer_times,refer_times,time_freq,sta_ids2,station_dict,l_data=l_data,n_data=n_data,GaWIN=GaWIN,GaWIN_flag=GaWIN_flag,R=R,R_flag=R_flag,RD=RD,RD_flag=RD_flag,Rxxday=Rxxday)
            refer_data[scene_a][insti_a]=cmip_result
            base_p[scene_a][insti_a]=cmip_result.iloc[:,1::].mean().to_frame().round(1).T.reset_index(drop=True).to_dict(orient='records')

    #%% 预估时段
    # 预估数据
    stats_start_year,stats_end_year=time_choose(time_freq,stats_times)
    
    pre_data=dict()
    for insti_a in insti:
        pre_data[insti_a]=dict()
        for scene_a in scene:
            pre_data[insti_a][scene_a]=dict()
            
            cmip_result=extreme_pre(element,data_dir,time_scale,insti_a,scene_a,nc_dict[ele_dict[element]],refer_times,stats_times,time_freq,sta_ids2,station_dict,l_data=l_data,n_data=n_data,GaWIN=GaWIN,GaWIN_flag=GaWIN_flag,R=R,R_flag=R_flag,RD=RD,RD_flag=RD_flag,Rxxday=Rxxday)
            pre_data[insti_a][scene_a]=cmip_result
   
    ##%% 增加一下 1.5℃和2.0℃
    if int(stats_end_year) >= 2020:
        
        refer_data['1.5℃']=dict()
        base_p['1.5℃']=dict()
        for insti_b,insti_b_table in pre_data.items():
            pre_data[insti_b]['1.5℃']=pre_data[insti_b]['ssp126'][(pre_data[insti_b]['ssp126']['年']>=2020) & (pre_data[insti_b]['ssp126']['年']<=2039)]
            
            refer_data['1.5℃'][insti_b]=refer_data['ssp126'][insti_b]
            base_p['1.5℃'][insti_b]=base_p['ssp126'][insti_b]
        
        scene=['ssp126','ssp245','ssp585','1.5℃']

    if int(stats_end_year) >= 2040:
        refer_data['2.0℃']=dict()
        base_p['2.0℃']=dict()
        for insti_b,insti_b_table in pre_data.items():
            pre_data[insti_b]['2.0℃']=pre_data[insti_b]['ssp245'][(pre_data[insti_b]['ssp245']['年']>=2040) & (pre_data[insti_b]['ssp245']['年']<=2059)]
            
            refer_data['2.0℃'][insti_b]=refer_data['ssp245'][insti_b]
            base_p['2.0℃'][insti_b]=base_p['ssp245'][insti_b]
            
        scene=['ssp126','ssp245','ssp585','1.5℃','2.0℃']
         
    #%% 单模式距平和距平百分率
    pre_data_result=dict()
    for i in insti:
        pre_data_result[i]=dict()
        for j in scene:
            pre_data_result[i][j]=dict()
            pre_data_result[i][j][element]=data_deal_2(pre_data[i][j],refer_data[j][i],2).to_dict(orient='records')

    #%% 结果保存
   
    result_df_dict=dict()
    result_df_dict['站点']=station_dict.to_dict(orient='records')
    result_df_dict['表格']=dict()

    result_df_dict['表格']['预估']=dict()
        
    for scene_a in scene:
        result_df_dict['表格']['预估'][scene_a]=dict()
        for insti_a in insti:
            result_df_dict['表格']['预估'][scene_a][insti_a]=dict()
            result_df_dict['表格']['预估'][scene_a][insti_a]=data_deal_2(pre_data[insti_a][scene_a],refer_data[scene_a][insti_a],1).to_dict(orient='records')


    result_df_dict['时序图']=dict()
    # result_df_dict['时序图']['集合_多模式' ]=dict()
    # result_df_dict['时序图']['集合_多模式' ]=percentile_std(['ssp126','ssp245','ssp585'],insti,pre_data,'none',refer_data)
    
    result_df_dict['时序图']['单模式' ]=pre_data_result
    #result_df_dict['时序图']['单模式' ]=dict()
    result_df_dict['时序图']['单模式' ]['基准期']=base_p
    
    #%% 外附：分布图绘制 1：实时绘图

    if plot == 1:
        # 观测绘图
        all_png = dict()
            
        # 预估
        all_png['预估']=dict()
        cmip_res=result_df_dict['表格']['预估']
        
        for exp, sub_dict1 in cmip_res.items():
            if exp in ['ssp126','ssp245','ssp585']:
                all_png['预估'][exp] = dict()
                for insti,stats_table in sub_dict1.items():
               
                    
                    all_png['预估'][exp][insti] = dict()
                    stats_table = pd.DataFrame(stats_table).iloc[:,:-5:]
                    stats_table=stats_table[['时间']+(station_dict['站号'].to_list())]
                    for i in range(len(stats_table)):
                        value_list = stats_table.iloc[i,1::]
                        year_name = stats_table.iloc[i,0]
                        exp_name = exp
                        insti_name = insti
                        # 插值/掩膜/画图/保存
                        mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                        png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, data_out)
                        
                        # 转url
                        png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                        png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
    
                        all_png['预估'][exp][insti][year_name] = png_path
    else:
        all_png=None
    
    result_df_dict['分布图']=all_png
    
    return result_df_dict

if __name__ == '__main__':
    
    data_json = dict()
    data_json['time_freq'] = 'Y'
    data_json['evaluate_times'] = "2018,2050" # 预估时段时间条
    data_json['refer_times'] = '1995,2014'# 参考时段时间条
    data_json['sta_ids'] = '51886,52602,52633,52645,52657,52707,52713'
    data_json['cmip_type'] = 'original' # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    data_json['cmip_res'] = None # 分辨率 1/5/10/25/50/100 km
    data_json['cmip_model'] = ["NESM3", "MPI-ESM1-2-LR", "RCM_BCC", "Set"]# 模式，列表：['CanESM5','CESM2']等
    data_json['element'] = "TN10p"
    data_json['l_data'] = 10
    data_json['GaWIN_flag'] = 3
    data_json['shp_path'] = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\州界.shp'
    data_json['plot'] = 1
    
    result=extreme_climate_esti(data_json)
