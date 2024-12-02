                                                                         # -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:06:27 2024

    参数说明：
    
    :param element: 采暖度日： HDD18;采暖日： HD;暖起始日：HTIME  没啥用，都会跑
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
from Module02.page_energy.wrapped.func00_function import choose_mod_path
from Module02.page_energy.wrapped.func00_function import time_choose
from Module02.page_energy.wrapped.func00_function import data_deal
from Module02.page_energy.wrapped.func00_function import data_deal_num
from Module02.page_energy.wrapped.func00_function import data_deal_2
from Module02.page_energy.wrapped.func00_function import data_deal_num_2
from Module02.page_energy.wrapped.func00_function import calculate_average_hd
from Module02.page_energy.wrapped.func00_function import percentile_std
from Module02.page_energy.wrapped.func00_function import percentile_std_time

from Module02.page_energy.wrapped.func01_winter_heating_pre import winter_heating_pre
from Module02.page_energy.wrapped.func02_winter_heating_his import winter_heating_his
from Module02.page_climate.wrapped.func03_plot import interp_and_mask, plot_and_save
from Utils.data_loader_with_threads import get_database_data

#%% main
def energy_winter_heating(data_json):

    
    #%% 参数信息
    element = data_json['element']
    refer_times = data_json['refer_times']
    time_freq = data_json['time_freq']
    stats_times = data_json['stats_times']
    sta_ids = data_json['sta_ids']
    data_cource = data_json['data_cource']
    insti = data_json['insti']
    res = data_json.get('res', '25')
    plot= data_json.get('plot')
    shp_path=data_json['shp_path']
    method='idw'
    #%% 固定信息
    
    # data_dir='/zipdata'
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
    # nc要素
    var='tas'
    
    # 情景选择
    # 'ssp126','ssp245','ssp585','1.5℃'，'2.0℃'
    scene=['ssp126','ssp245','ssp585']
    
    # 时间频率
    time_scale='daily'
    
    elem_info=['采暖度日','采暖日','采暖起始日_日序']
    elem_dict=dict()
    elem_dict['采暖日']='HD'
    elem_dict['采暖度日']='HDD18'
    elem_dict['采暖起始日_日序']='HDTIME_NUM'
    
    uuid4 = uuid.uuid4().hex
    data_out = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_out):
        os.makedirs(data_out)
        os.chmod(data_out, 0o007 | 0o070 | 0o700)
    #%% 统计计算模块
    
    # 评估数据
    elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,TEM_Avg'
    sta_ids1 = tuple(sta_ids.split(','))
    
    
    refer_df = get_database_data(sta_ids1, elements, 'qh_qhbh_cmadaas_day', time_freq, refer_times)

    refer_df.set_index('Datetime', inplace=True)
    refer_df.index = pd.DatetimeIndex(refer_df.index)
    refer_df['Station_Id_C'] = refer_df['Station_Id_C'].astype(str)
    refer_df.sort_index(inplace=True)
    
    if 'Unnamed: 0' in refer_df.columns:
        refer_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    refer_result_days,refer_result_hdd18,refer_result_start_end,refer_result_start_end_num= winter_heating_his(refer_df)
        
    refer_result_days_z=data_deal(refer_result_days)
    refer_result_hdd18_z=data_deal(refer_result_hdd18)
    refer_result_start_end_num_z=data_deal_num(refer_result_start_end_num)
    
    # 加一个 站点站名字典
    station_id=refer_df['Station_Id_C'].unique()
    
    matched_stations = pd.merge(pd.DataFrame({'Station_Id_C': station_id}),refer_df[['Station_Id_C', 'Station_Name','Lon','Lat']],on='Station_Id_C')
    matched_stations_unique = matched_stations.drop_duplicates(subset='Station_Id_C')

    station_name = matched_stations_unique['Station_Name'].values
    station_id=matched_stations_unique['Station_Id_C'].values
    station_dict=pd.DataFrame(columns=['站名','站号'])
    station_dict['站名']=station_name
    station_dict['站号']=station_id
    lon_list=matched_stations_unique['Lon'].values
    lat_list=matched_stations_unique['Lat'].values

    
    # 预估数据
    insti = insti.split(',')
    sta_ids2=sta_ids.split(',')
    
    pre_data=dict()
    for insti_a in insti:
        pre_data[insti_a]=dict()
        for scene_a in scene:
            pre_data[insti_a][scene_a]=dict()
            result_days,result_hdd18,result_start_end,result_start_end_num= winter_heating_pre(element,data_dir,time_scale,insti_a,scene_a,var,stats_times,time_freq,sta_ids2,station_dict)
    
            pre_data[insti_a][scene_a]['HDD18']=result_hdd18
            pre_data[insti_a][scene_a]['HD']=result_days
            pre_data[insti_a][scene_a]['HDTIME']=result_start_end
            pre_data[insti_a][scene_a]['HDTIME_NUM']=result_start_end_num
        
#%% 基准期    
    base_p=pd.DataFrame(columns=refer_result_days_z.columns[1:-3:])
    base_p.loc[0,:]=refer_result_days_z[refer_result_days_z['时间'] == '平均'][1::-3].iloc[0, :]
    base_p.loc[1,:]=refer_result_hdd18_z[refer_result_hdd18_z['时间'] == '平均'][1::-3].iloc[0, :]
    base_p.loc[2,:]=refer_result_start_end_num_z[refer_result_start_end_num_z['时间'] == '平均'][1::-3].iloc[0, :]
    base_p.insert(0, '要素', ['采暖日','采暖度日','采暖起始日_日序'])
    
#%% 单模式距平和距平百分率

    pre_data_result=dict()
    for i in insti:
        if i != 'Set':
            pre_data_result[i]=dict()
            for j in scene:
                pre_data_result[i][j]=dict()
                for ele_a in elem_info:
                    if ele_a in ['采暖度日','采暖日']:
                        pre_data_result[i][j][ele_a]=data_deal_2(pre_data[i][j][elem_dict[ele_a]],refer_result_days,2).to_dict(orient='records')
                    else:
                        pre_data_result[i][j][ele_a]=data_deal_num_2(pre_data[i][j][elem_dict[ele_a]],refer_result_days,2).to_dict(orient='records')

#%% 结果保存
   
    result_df_dict=dict()
    result_df_dict['站点']=station_dict.to_dict(orient='records')
    result_df_dict['表格']=dict()

    result_df_dict['表格']['历史']=dict()
    result_df_dict['表格']['历史']['采暖日']=refer_result_days_z.to_dict(orient='records')
    result_df_dict['表格']['历史']['采暖度日']=refer_result_hdd18_z.to_dict(orient='records')
    result_df_dict['表格']['历史']['采暖起始日_日期']=refer_result_start_end.to_dict(orient='records')
    result_df_dict['表格']['历史']['采暖起始日_日序']=refer_result_start_end_num_z.to_dict(orient='records')
    
    result_df_dict['表格']['预估']=dict()
        
    for scene_a in scene:
        result_df_dict['表格']['预估'][scene_a]=dict()
        for insti_a in insti:
            result_df_dict['表格']['预估'][scene_a][insti_a]=dict()
            result_df_dict['表格']['预估'][scene_a][insti_a]['采暖日']=data_deal_2(pre_data[insti_a][scene_a]['HD'],refer_result_days,1).to_dict(orient='records')
            result_df_dict['表格']['预估'][scene_a][insti_a]['采暖度日']=data_deal_2(pre_data[insti_a][scene_a]['HDD18'],refer_result_hdd18,1).to_dict(orient='records')
            result_df_dict['表格']['预估'][scene_a][insti_a]['采暖起始日_日期']=pre_data[insti_a][scene_a]['HDTIME'].to_dict(orient='records')
            result_df_dict['表格']['预估'][scene_a][insti_a]['采暖起始日_日序']=data_deal_num_2(pre_data[insti_a][scene_a]['HDTIME_NUM'],result_start_end_num,1).to_dict(orient='records')

    result_df_dict['时序图']=dict()
    result_df_dict['时序图']['集合_多模式' ]=dict()
    result_df_dict['时序图']['集合_多模式' ]['采暖日']=percentile_std(scene,insti,pre_data,'HD',refer_result_days)
    result_df_dict['时序图']['集合_多模式' ]['采暖度日']=percentile_std(scene,insti,pre_data,'HDD18',refer_result_hdd18)
    result_df_dict['时序图']['集合_多模式' ]['采暖起始日_日序']=percentile_std_time(scene,insti,pre_data,result_start_end_num)
    
    result_df_dict['时序图']['单模式' ]=pre_data_result
    result_df_dict['时序图']['单模式' ]['基准期']=base_p.to_dict(orient='records').copy()
    
    #%% 外附：分布图绘制 1：实时绘图
    def find_keys_by_value(d, value):
        return [key for key, val in d.items() if val == value]

    if plot == 1:
        # 观测绘图
        all_png = dict()

        # all_png['历史']=dict()
        # data_pic=pd.DataFrame(result_df_dict['表格']['历史'][find_keys_by_value(elem_dict, element)[0]]).iloc[:,:-3:]
        # for i in np.arange(len(data_pic)):
        #     mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, data_pic.iloc[i,1::], method)
        #     png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, '历史', '观测', str(data_pic.iloc[i,0]), data_out)
                    
        #     png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
        #     png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        #     all_png['历史'][str(data_pic.iloc[i,0])] = png_path
            
        # 预估
        all_png['预估']=dict()
        cmip_res=result_df_dict['表格']['预估']
        
        for exp, sub_dict1 in cmip_res.items():
            all_png['预估'][exp] = dict()
            for insti,stats_table in sub_dict1.items():
                all_png['预估'][exp][insti] = dict()
                stats_table = pd.DataFrame(stats_table[find_keys_by_value(elem_dict, element)[0]]).iloc[:,:-5:]
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
    data_json['element'] ='HDD18'
    data_json['refer_times'] = '2000,2010'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2020,2040'
    data_json['sta_ids'] = '52754,56151,52855,52862,56065,52645,56046,52955,52968,52963,52825,56067,52713,52943,52877,52633,52866'
    data_json['data_cource'] = 'original'
    data_json['insti'] = 'Set'
    # data_json['res'] ='1'
    data_json['shp_path'] = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\州界.shp'
    data_json['plot'] = 1

    
    result_df_dict=energy_winter_heating(data_json)
    
    result_example=pd.DataFrame( result_df_dict['时序图']['单模式' ]['基准期'])
