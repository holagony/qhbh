# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:52:09 2025

@author: EDY
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:37:23 2024

@author: EDY


参数说明：

:param element: 
    净初级生产力：NPP
    
:param refer_times: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'
:param time_freq: 对应原型选择数据的时间尺度
    传参：
    年 - 'Y'
    季 - 'Q'
    月(连续) - 'M1'
    月(区间) - 'M2' 
    
 :param stats_times: 对应原型的统计时段
     (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'
     (2)当time_freq选择季Q。下载连续的月数据，处理成季数据（多下两个月数据），提取历年相应的季节数据，传：['%Y,%Y','3,4,5']，其中'3,4,5'为季度对应的月份 
     (3)当time_freq选择月(连续)M1。下载连续的月数据，传参：'%Y%m,%Y%m'
     (4)当time_freq选择月(区间)M2。下载连续的月数据，随后提取历年相应的月数据，传参：['%Y,%Y','11,12,1,2'] 前者年份，'11,12,1,2'为连续月份区间

 :param refer_times: 和stats_times格式相同
    
 :param sta_ids: 传入的站点，多站，传：'52866,52713,52714'   
 :param data_cource: 预估数据源，传：'original'， 'Delta'， 'RDF' ，'RF' 
 :param insti: 模式选择，传：'BCC-CSM2-MR,CanESM5' 
 :param res 分辨率 ，数据源为original时可不传
    传参：
    25 - '25' 
    50 - '50'
    100 - '100'

"""

import numpy as np
import os
import pandas as pd
import uuid
from Utils.config import cfg
from Module02.page_energy.wrapped.func00_function import data_deal_2
from Module02.page_climate.wrapped.func_plot import interp_and_mask, plot_and_save
from Module02.page_grass.wrapped.func04_cesva_pre import cesva_pre


#%% main
def grass_cesva(data_json):

    
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
    
    # 时间频率
    time_scale='daily'
    
    # 变量
    if element=='TR':
        var='rsds'
    else:
        var='sund'
    
    # 情景选择
    scene=['ssp126','ssp245','ssp585']
    
    elem_dict=dict()
    elem_dict['NPP']='净初级生产力'

    uuid4 = uuid.uuid4().hex
    data_out = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_out):
        os.makedirs(data_out)
        os.chmod(data_out, 0o007 | 0o070 | 0o700)
    
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

    insti = insti.split(',')
    sta_ids2=sta_ids.split(',')    

    #%% 统计计算模块 基准期
    refer_data=dict()
    base_p=dict()
    for scene_a in scene:
        refer_data[scene_a]=dict()
        base_p[scene_a]=dict()

        for insti_a in insti:
                refer_data[scene_a][insti_a]=dict()
                base_p[scene_a][insti_a]=dict()
                base_p[scene_a][insti_a][elem_dict[element]]=dict()
                
                result= cesva_pre(element,data_dir,time_scale,insti_a,scene_a,refer_times,time_freq,sta_ids2,station_dict)
                refer_data[scene_a][insti_a]=result
                base_p[scene_a][insti_a][elem_dict[element]]=result.iloc[:,1::].mean().to_frame().round(1).T.reset_index(drop=True).to_dict(orient='records')

    
    #%% 预估数据
    pre_data=dict()
    for insti_a in insti:
        pre_data[insti_a]=dict()
        for scene_a in scene:
            pre_data[insti_a][scene_a]=dict()
            
            result= cesva_pre(element,data_dir,time_scale,insti_a,scene_a,stats_times,time_freq,sta_ids2,station_dict)
            pre_data[insti_a][scene_a]=result
            
    stats_end_year=result['年'].iloc[-1]
    
    ##%% 增加一下 1.5℃和2.0℃
    if int(stats_end_year) >= 2020:
        refer_data['1.5℃']=dict()
        base_p['1.5℃']=dict()

        for insti_b,insti_b_table in pre_data.items():
            pre_data[insti_b]['1.5℃']=pre_data[insti_b]['ssp126'][(pre_data[insti_b]['ssp126']['年'].astype(int)>=2020) & (pre_data[insti_b]['ssp126']['年'].astype(int)<=2039)]
            refer_data['1.5℃'][insti_b]=refer_data['ssp126'][insti_b]
            base_p['1.5℃'][insti_b]=base_p['ssp126'][insti_b]
        
        scene=['ssp126','ssp245','ssp585','1.5℃']

    if int(stats_end_year) >= 2040:
        refer_data['2.0℃']=dict()
        base_p['2.0℃']=dict()
        for insti_b,insti_b_table in pre_data.items():
            pre_data[insti_b]['2.0℃']=pre_data[insti_b]['ssp245'][(pre_data[insti_b]['ssp245']['年'].astype(int)>=2040) & (pre_data[insti_b]['ssp245']['年'].astype(int)<=2059)]
     
            refer_data['2.0℃'][insti_b]=refer_data['ssp245'][insti_b]
            base_p['2.0℃'][insti_b]=base_p['ssp245'][insti_b]
   
        scene=['ssp126','ssp245','ssp585','1.5℃','2.0℃']
        

    #%% 单模式距平和距平百分率
    pre_data_result=dict()
    for i in insti:
        pre_data_result[i]=dict()
        for j in scene:
            pre_data_result[i][j]=dict()
            pre_data_result[i][j][elem_dict[element]]=data_deal_2(pre_data[i][j],refer_data[j][i],1)
    
    #%% 结果保存   
    result_df=dict()
    result_df['站点']=station_dict.to_dict(orient='records')
    result_df['表格']=dict()
    result_df['表格']['预估']=dict()
    for scene_a in scene:
        result_df['表格']['预估'][scene_a]=dict()
        for insti_a in insti:
            result_df['表格']['预估'][scene_a][insti_a]=dict()
            result_df['表格']['预估'][scene_a][insti_a][elem_dict[element]]=data_deal_2(pre_data[insti_a][scene_a],refer_data[scene_a][insti_a],1).to_dict(orient='records')

    result_df['时序图']=dict()
    # result_df['时序图']['集合_多模式' ]=dict()
    # result_df['时序图']['集合_多模式' ]=percentile_std(['ssp126','ssp245','ssp585'],insti,pre_data,'none',refer_result)
    
    result_df['时序图']['单模式' ]=pre_data_result.copy()
    result_df['时序图']['单模式' ]['基准期']=base_p#.to_dict(orient='records').copy()

   
    def convert_dataframes_to_dicts(multi_dict):
        """
        递归地将多重字典中的DataFrame转换为字典。
        """
        converted_dict = {}
        for key, value in multi_dict.items():
            if isinstance(value, pd.DataFrame):
                converted_dict[key] = value.to_dict(orient='records')
            elif isinstance(value, dict):
                converted_dict[key] = convert_dataframes_to_dicts(value)
            else:
                converted_dict[key] = value
        return converted_dict

    result_df=convert_dataframes_to_dicts(result_df)
    
    
    if plot == 1:
        # 观测绘图
        all_png = dict()
 
        # 预估
        all_png['预估']=dict()
        cmip_res=result_df['表格']['预估']
        
        for exp, sub_dict1 in cmip_res.items():
            if exp in ['ssp126','ssp245','ssp585']:
                all_png['预估'][exp] = dict()
                for insti,stats_table in sub_dict1.items():
                    all_png['预估'][exp][insti] = dict()
                    stats_table = pd.DataFrame(stats_table[elem_dict[element]]).iloc[:,:-5:]
                    stats_table=stats_table[['时间']+(station_dict['站号'].to_list())]
                    
                    for i in range(len(stats_table)):
                        value_list = stats_table.iloc[i,1::]
                        year_name = stats_table.iloc[i,0]
                        exp_name = exp
                        insti_name = insti
                        # 插值/掩膜/画图/保存
                        
                        try:
                            mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
                            png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, data_out)
                            
                            # 转url
                            png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
                            png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
         
                            all_png['预估'][exp][insti][year_name] = png_path
                        except:
                            all_png['预估'][exp][insti][year_name] =None
    else:
        all_png=None
   
    result_df['分布图']=all_png
    
    return result_df
    
if __name__ == '__main__':
    
    # 总辐射量：TR 
    # 直接辐射占比： PDR;
    # 日照时数：SH；
    # 有效日照天数：ASD    
    data_json = dict()
    data_json['element'] ='TR'
    data_json['refer_times'] = '1995,2024'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2025,2100'
    data_json['sta_ids'] = "51886,52602,52633,52645,52657,52707,52713,52737,52745,52754,52765,52818,52825,52833,52836,52842,52853,52855,52856,52862,52863,52866,52868,52869,52874,52875,52876,52877,52908,52943,52955,52957,52963,52968,52972,52974,56004,56016,56018,56021,56029,56033,56034,56043,56045,56046,56065,56067,56125,56151"
    data_json['data_cource'] = 'original'
    data_json['insti'] = 'Set'
    # data_json['res'] ='1'
    data_json['shp_path'] = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\州界.shp'
    data_json['plot'] = 1

