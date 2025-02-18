# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:40:25 2024

@author: Daimu

"""

import pandas as pd
import numpy as np
import os
from psycopg2 import sql
import psycopg2
from Utils.config import cfg
import re
import uuid
from sklearn.linear_model import LinearRegression
from Module02.page_ice.wrapped.func01_factor_data_deal import factor_data_deal
from Module02.page_ice.wrapped.func03_model_factor_data_deal import model_factor_data_deal
from Module02.page_energy.wrapped.func00_function import percentile_std
from Module02.page_climate.wrapped.func_plot import interp_and_mask, plot_and_save


from Module02.page_water.wrapped.water_evaluate_data_deal import water_evaluate_data_deal



def clean_column_name(name):
    # 替换空格和特殊字符为下划线
    cleaned_name = re.sub(r'\W+', '_', name)
    # 确保列名不以数字开头
    if cleaned_name[0].isdigit():
        cleaned_name = '_' + cleaned_name
    return cleaned_name

def trend_rate(x):
    '''
    计算变率（气候倾向率）的pandas apply func
    '''
    try:
        x = x.to_frame()
        x['num'] = x.index.tolist()
        x['num'] = x['num'].map(int)
        x.dropna(how='any', inplace=True)
        train_x = x.iloc[:, -1].values.reshape(-1, 1)
        train_y = x.iloc[:, 0].values.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
        weight = model.coef_[0][0].round(3) * 10
        return weight
    except:
        return np.nan
            
def data_deal_2(data_df,refer_df,flag):

    if '年' in data_df.columns:
        data_df.set_index(data_df['年'],inplace=True)
        data_df.drop(['年'], axis=1, inplace=True) 
        
    if flag == 2 or flag==3:
        a=data_df['区域均值'].copy()
        data_df.drop(['区域均值'], axis=1, inplace=True) 
        
    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = np.round(data_df.iloc[:, :].mean(axis=0).astype(float),2)
    tmp_df.loc['变率'] = np.round(data_df.apply(trend_rate, axis=0),2)
    tmp_df.loc['最大值'] = data_df.iloc[:, :].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[:, :].min(axis=0)
    tmp_df.loc['参考时段均值'] =  np.round(refer_df.iloc[:, :].mean(axis=0).astype(float),2)
    tmp_df.loc['距平'] =  np.round((tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']).astype(float),2)
    tmp_df.loc['距平百分率'] =  np.round(((tmp_df.loc['距平'] / tmp_df.loc['参考时段均值']) * 100).astype(float),2)

    # 合并所有结果
    stats_result = data_df.copy()
    if flag==1:
        stats_result['区域均值'] = np.round(data_df.iloc[:, :].mean(axis=1).astype(float),2)
    elif flag ==2 or flag==3:
        stats_result['区域均值']=a
    stats_result['区域距平'] = np.round((data_df.iloc[:, :].mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).astype(float),2)
    stats_result['区域距平百分率'] = np.round(((stats_result['区域距平']/tmp_df.loc['参考时段均值'].mean())*100).astype(float),2)
    stats_result['区域最大值'] = data_df.iloc[:, :].max(axis=1)
    stats_result['区域最小值'] = data_df.iloc[:, :].min(axis=1)

    if flag==3:
        stats_days_result = stats_result
    else:
        stats_days_result = pd.concat((stats_result, tmp_df), axis=0)

    stats_days_result.insert(loc=0, column='时间', value=stats_days_result.index)
    stats_days_result.reset_index(drop=True, inplace=True)
    
    return stats_days_result    

    
def water_table_def(data_json):

    main_element=data_json['main_element']
    hydro_ids = data_json['hydro_ids']
    #sta_ids=data_json['sta_ids']
    time_freq_main=data_json['time_freq_main']
    time_freq_main_data=data_json['time_freq_main_data']
    refer_times=data_json['refer_times']
    stats_times=data_json['stats_times']
    data_cource=data_json['data_cource']
    insti=data_json['insti']
    res = data_json.get('res', '10')
    factor_element=data_json['factor_element']
    factor_time_freq=data_json['factor_time_freq']
    factor_time_freq_data=data_json['factor_time_freq_data']
    verify_time=data_json['verify_time']

    plot= data_json.get('plot')
    shp_path=data_json['shp_path']
    method='idw'
    
    time_scale='daily'
    
    if shp_path is not None:
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径
        
    scene=['ssp126','ssp245','ssp585']
    independent_columns=factor_element.split(',')
    
    elements=factor_element.split(',')
    time_freqs =factor_time_freq.split(',')
    
    factor_name = []
    for i in range(len(elements)):
        combined_str = f"{elements[i]}_{time_freqs[i]}_{factor_time_freq_data[i]}"
        cleaned_name = clean_column_name(combined_str)
        factor_name.append(cleaned_name)
        
    #%% 固定字典表
    if os.name == 'nt':
        data_dir=r'D:\Project\qh\hydrological_station\original'
    elif os.name == 'posix':
        data_dir='/model_data/hydrological_station/original'
    else:
        data_dir='/model_data/hydrological_station/original'

    if data_cource == 'original':
        res='25'
        
    res_d=dict()
    res_d['25']='0.25deg'
    res_d['50']='0.50deg'
    res_d['100']='1.00deg'
   
    if data_cource != 'original':
        data_dir=os.path.join('/model_data/hydrological_station/Delta',res_d[res])
        
    # 模型站点要素名对照表
    model_ele_dict=dict()
    model_ele_dict['TEM_Avg']='tas'
    model_ele_dict['PRE_Time_2020']='pr'
    model_ele_dict['WIN_S_2mi_Avg']='ws'
    model_ele_dict['RHU_Avg']='hurs'
    model_ele_dict['TEM_Min']='tasmin'
    model_ele_dict['TEM_Max']='tasmax'

        
    # 不同的要素不同的处理方法,针对日尺度之外的处理
    resample_max = ['TEM_Max', 'PRS_Max', 'WIN_S_Max', 'WIN_S_Inst_Max', 'GST_Max', 'huangku']

    resample_min = ['TEM_Min', 'PRS_Min', 'GST_Min', 'RHU_Min', 'fanqing']

    resample_sum = ['SSH','PRE_Time_2020', 'PRE_Days', 'EVP_Big', 'EVP', 'EVP_Taka', 'PMET','sa','rainstorm','light_snow','snow',
                    'medium_snow','heavy_snow','severe_snow','Hail_Days','Hail','GaWIN',
                    'GaWIN_Days','SaSt','SaSt_Days','FlSa','FlSa_Days','FlDu','FlDu_Days',
                    'Thund','Thund_Days''high_tem','drought','light_drought','medium_drought',
                    'heavy_drought','severe_drought','Accum_Tem']

    resample_mean = ['TEM_Avg', 'PRS_Avg', 'WIN_S_2mi_Avg', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Avg_5cm', 'GST_Avg_10cm', 
                     'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 
                     'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg', 'Cov', 'pmet']

    processing_methods = {element: 'mean' for element in resample_mean}
    processing_methods.update({element: 'sum' for element in resample_sum})
    processing_methods.update({element: 'max' for element in resample_max})
    processing_methods.update({element: 'min' for element in resample_min})
    


    uuid4 = uuid.uuid4().hex
    data_out = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_out):
        os.makedirs(data_out)
        os.chmod(data_out, 0o007 | 0o070 | 0o700)
    #%% 站点字典表
    station_path=os.path.join(data_dir,time_scale,'Set/historical/tas.csv')
    df_station=pd.read_csv(station_path)
    df_station=df_station.iloc[0:3:,:].T.reset_index()
    df_station.columns = ['站点名','区站号', '纬度','经度']
    df_station = df_station.drop(0)
    
    df_station['区站号']=df_station['区站号'].astype(int).astype(str)
    
    station_id=hydro_ids.split(',')
    
    matched_stations = pd.merge(pd.DataFrame({'区站号': station_id}),df_station[['区站号', '站点名','纬度','经度']],on='区站号')
    matched_stations_unique = matched_stations.drop_duplicates()
    
    station_name = matched_stations_unique['站点名'].values
    station_id=matched_stations_unique['区站号'].values
    station_dict=pd.DataFrame(columns=['站名','站号'])
    station_dict['站名']=np.array([s[:s.find('国')] if '国' in s else s for s in station_name])
    station_dict['站号']=station_id
    lon_list=matched_stations_unique['经度'].values
    lat_list=matched_stations_unique['纬度'].values 
    
    station_id_c=hydro_ids.split(',')
    #%% 表格数据
    # 参考时段
    all_png=dict()
    
    # 模拟模式
    var_factor_time_freq=factor_time_freq.split(',')
    instis=insti.split(',')
    sta_ids2=hydro_ids.split(',')
    
    stats_start_year=refer_times.split(',')[0]
    stats_end_year=refer_times.split(',')[1]
    years_len=np.arange(int(stats_start_year),int(stats_end_year)+1)
        
    # 单模式单情景
    refer_data_1=dict()
    base_p=dict()
    for scene_a in scene:
        refer_data_1[scene_a]=dict()
        base_p[scene_a]=dict()
        for insti_a in instis:
    
            refer_data_1[scene_a][insti_a]=dict()
            base_p[scene_a][insti_a]=dict()

            data_station_dataframes = []
            for index,var_a in enumerate(independent_columns):
                data_station=model_factor_data_deal(data_dir, time_scale,insti_a,scene_a,sta_ids2,model_ele_dict[var_a],var_a,var_factor_time_freq[index],factor_time_freq_data[index],time_freq_main,refer_times,processing_methods)
                data_station = data_station.sort_values(by=['Station_Id_C', '年'], ascending=[True, True])
                data_station=data_station.rename(columns={var_a:factor_name[index]})
                data_station_dataframes.append(data_station)
            b=pd.concat(data_station_dataframes, axis=1)
            b = b.loc[:, ~b.columns.duplicated()]
            
            grouped = b.groupby('年')
            group_averages = grouped[factor_name].mean()
            
            result_4=pd.DataFrame(index=np.arange(len(years_len)),columns=station_id_c)
            for i in station_id_c:
                station_i=b[b['Station_Id_C']==i]
                station_i = station_i[factor_name].astype(float).reset_index(drop=True)
                result_4[i] =(data_json['intercept'] + np.sum(station_i * pd.Series(data_json)[factor_name],axis=1)).astype(float).round(2)
            result_4.insert(0, '年', years_len)
            result_4[result_4<0]=0  
            refer_data_1[scene_a][insti_a]=result_4.copy()
            base_p[scene_a][insti_a]=result_4.iloc[:,1::].mean().to_frame().round(1).T.reset_index(drop=True).to_dict(orient='records')

    
    #%% 评估数据
    stats_start_year=stats_times.split(',')[0]
    stats_end_year=stats_times.split(',')[1]
    years_len=np.arange(int(stats_start_year),int(stats_end_year)+1)
        
    # 单模式单情景
    pre_data_4=dict()
    pre_data_5=dict()
    for insti_a in instis:
        pre_data_4[insti_a]=dict()
        pre_data_5[insti_a]=dict()

        for scene_a in scene:
            pre_data_4[insti_a][scene_a]=dict()
            pre_data_5[insti_a][scene_a]=dict()

            data_station_dataframes = []
            for index,var_a in enumerate(independent_columns):
                data_station=model_factor_data_deal(data_dir, time_scale,insti_a,scene_a,sta_ids2,model_ele_dict[var_a],var_a,var_factor_time_freq[index],factor_time_freq_data[index],time_freq_main,stats_times,processing_methods)
                data_station = data_station.sort_values(by=['Station_Id_C', '年'], ascending=[True, True])
                data_station=data_station.rename(columns={var_a:factor_name[index]})
                data_station_dataframes.append(data_station)
            b=pd.concat(data_station_dataframes, axis=1)
            b = b.loc[:, ~b.columns.duplicated()]
            
            grouped = b.groupby('年')
            group_averages = grouped[factor_name].mean()
            
            result_4=pd.DataFrame(index=np.arange(len(years_len)),columns=station_id_c)
            for i in station_id_c:
                station_i=b[b['Station_Id_C']==i]
                station_i = station_i[factor_name].astype(float).reset_index(drop=True)
                result_4[i] =(data_json['intercept'] + np.sum(station_i * pd.Series(data_json)[factor_name],axis=1)).astype(float).round(2)
            result_4.insert(0, '年', years_len)
            result_4[result_4<0]=0                     
            group_averages_i=group_averages[factor_name].astype(float).reset_index(drop=True)
            result_4['区域均值']=(data_json['intercept'] + np.sum(group_averages_i * pd.Series(data_json)[factor_name],axis=1)).astype(float).round(2)
           
            pre_data_5[insti_a][scene_a][main_element]=dict()
            pre_data_5[insti_a][scene_a][main_element]=result_4.copy()
            
            result_4_1=data_deal_2(result_4,refer_data_1[scene_a][insti_a],2)
            pre_data_4[insti_a][scene_a]=result_4_1.to_dict(orient='records')
   
    ##%% 增加一下 1.5℃和2.0℃
    if int(stats_end_year) >= 2020:
        
        refer_data_1['1.5℃']=dict()
        base_p['1.5℃']=dict()
        for insti_b,insti_b_table in pre_data_5.items():
            refer_data_1['1.5℃'][insti_b]=refer_data_1['ssp126'][insti_b]
            base_p['1.5℃'][insti_b]=base_p['ssp126'][insti_b]
            
            pre_data_5[insti_b]['1.5℃']=dict()
            pre_data_5[insti_b]['1.5℃'][main_element]=dict()
            pre_data_5[insti_b]['1.5℃'][main_element]=insti_b_table['ssp126'][main_element][(insti_b_table['ssp126'][main_element]['年']>=2020) & (insti_b_table['ssp126'][main_element]['年']<=2039)]
            result_4_1=data_deal_2(pre_data_5[insti_b]['1.5℃'][main_element].copy(),refer_data_1['1.5℃'][insti_b],2)
            pre_data_4[insti_b]['1.5℃']=result_4_1.to_dict(orient='records')
              
        scene=['ssp126','ssp245','ssp585','1.5℃']

    if int(stats_end_year) >= 2040:
        
        refer_data_1['2.0℃']=dict()
        base_p['2.0℃']=dict()
        for insti_b,insti_b_table in pre_data_5.items():
            refer_data_1['2.0℃'][insti_b]=refer_data_1['ssp245'][insti_b]
            base_p['2.0℃'][insti_b]=base_p['ssp245'][insti_b]  
            
            pre_data_5[insti_b]['2.0℃']=dict()
            pre_data_5[insti_b]['2.0℃'][main_element]=dict()
            pre_data_5[insti_b]['2.0℃'][main_element]=insti_b_table['ssp245'][main_element][(insti_b_table['ssp245'][main_element]['年']>=2040) & (insti_b_table['ssp245'][main_element]['年']<=2059)]
            result_4_1=data_deal_2(pre_data_5[insti_b]['2.0℃'][main_element].copy(),refer_data_1['2.0℃'][insti_b],2)
            pre_data_4[insti_b]['2.0℃']=result_4_1.to_dict(orient='records')
          
        scene=['ssp126','ssp245','ssp585','1.5℃','2.0℃']    
    #%% 时序图  
    pre_data_6=dict()
    for i in instis:
        pre_data_6[i]=dict()
        for j in scene:
            pre_data_6[i][j]=data_deal_2(pre_data_5[i][j][main_element],refer_data_1[j][i],3).to_dict(orient='records')
    
                
    #%% 保存
    result_df_dict=dict()
    result_df_dict['表格']=dict()
    result_df_dict['表格']['预估']=dict()

    for exp, sub_dict1 in pre_data_4.items():
        for insti,stats_table in sub_dict1.items():
            if insti not in result_df_dict['表格']['预估']:
                result_df_dict['表格']['预估'][insti]=dict()
            
            if main_element=='Q':
                stats_table_result=pd.DataFrame()
                stats_table=pd.DataFrame(stats_table)
                stats_table_result['时间']=stats_table['时间']
                stats_table_result['站名']=station_dict['站名'].values[0]
                stats_table_result['站号']=station_dict['站号'].values[0]
                stats_table_result['Q']=stats_table[station_dict['站号']]
                stats_table_result['距平']=stats_table['区域距平']
                stats_table_result['距平百分率']=stats_table['区域距平百分率']
                stats_table_result=stats_table_result.iloc[:-7:,:]
                result_df_dict['表格']['预估'][insti][exp]=stats_table_result.to_dict(orient='records')
            else:
                result_df_dict['表格']['预估'][insti][exp]=stats_table

            
    result_df_dict['时序图']=dict()    
    result_df_dict['时序图']['单模式' ]=pre_data_6
    result_df_dict['时序图']['单模式' ]['基准期']=base_p
    result_df_dict['站点']=station_dict.to_dict(orient='records')

    
    if plot==1:

        
        # 预估
        all_png['预估']=dict()
        cmip_res=result_df_dict['表格']['预估']
        
        for exp, sub_dict1 in cmip_res.items():
            if exp in ['ssp126','ssp245','ssp585']:

                all_png['预估'][exp] = dict()
                for insti,stats_table in sub_dict1.items():
                    all_png['预估'][exp][insti] = dict()
                    stats_table = pd.DataFrame(stats_table).iloc[:,:-5:]
                    
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
                
                
     
        
    result_df_dict['分布图']=all_png
        
        
        
    return result_df_dict
    
if __name__ == '__main__':
    
    
    data_json = dict()
    data_json['main_element']='Q'  # 评估要素
    data_json['hydro_ids']='40100350' # 站点信息
    data_json['sta_ids']='52943,56021,56045,56065' # 站点信息
    data_json['time_freq_main']='Y' # 评估要素时间尺度
    data_json['time_freq_main_data']='0'
    data_json['refer_times'] = '2020,2024' # 参考时段
    data_json['stats_times'] = '2020,2040' # 评估时段
    data_json['data_cource'] = 'original' # 模式信息
    data_json['insti']= 'Set'
    data_json['res'] ="None"
    data_json['factor_element']='TEM_Avg,PRE_Time_2020,TEM_Avg'     # 关键因子
    data_json['factor_time_freq']='Y,Q,M2' # 关键因子时间尺度
    data_json['factor_time_freq_data']=['0','3,4,5','1']
    data_json['verify_time']='2020,2024' # 验证日期

    # 要素变量
    data_json['intercept']=1
    data_json['TEM_Avg_Y_0']=2
    data_json['PRE_Time_2020_Q_3_4_5']=3
    data_json['TEM_Avg_M2_1']=3


    # 分布图
    data_json['shp_path'] = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\州界.shp'
    data_json['plot'] = 0
    
    
    result=water_table_def(data_json)
    