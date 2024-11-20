# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:53:36 2024

@author: EDY
"""

import time
import pandas as pd
import psycopg2
from Utils.config import cfg
from psycopg2 import sql
from sklearn.linear_model import LinearRegression
import numpy as np
from pandas.tseries.offsets import MonthEnd

def agriculture_features_stats(data_json):
    '''
    获取天擎数据，参数说明
    :param element：对应原型，传入的要素名称
        作物种植面积：crop_acreage
        产量：yield
        播种期：11 sowin_date
        成熟期：91 maturity:
        生育期：reproductive_period
        生育期天数：reproductive_day
        
    :param crop: 对应原型，作物
        春玉米中熟：010402 spring_maizet
        春小麦：010304 spring_wheat
        冬小麦：010301 winter_wheat
        青稞：010307 highland_barley
        油菜甘蓝型：010603 napus
        油菜白菜型：010602 campestris
        蚕豆：010907 Horsebean
        马铃薯：010906 potato

    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'
        有效时间： 2010 - 2024
    :param stats_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'

    :param sta_ids: 传入的站点，多站，传：'52866,52713,52714'

    :param interp_method: 对应原型的插值方法
        传参：
        克里金 - 'kriging'
        泛克里金 - 'uni_kriging'
        反距离权重 - 'idw'

    :param ci: 置信区间    
    :param shp_path: shp文件
    :param output_filepath: 输出结果文件

    '''
    # 1.参数读取
    element = data_json['element']
    crop = data_json['crop']
    refer_years = data_json['refer_years']
    nearly_years = data_json['nearly_years']
    stats_times = data_json['stats_times']
    sta_ids = data_json['sta_ids']
    last_year = int(nearly_years.split(',')[-1])  # 上一年的年份

    # 2.参数处理
    # 确定表名
    table_dict = dict()
    table_dict['crop_acreage'] = '待定'
    table_dict['yield'] = '待定'
    table_dict['sowin_date'] = 'qh_climate_crop_growth'
    table_dict['maturity'] = 'qh_climate_crop_growth'
    table_dict['reproductive_period'] = 'qh_climate_crop_growth'
    table_dict['reproductive_day'] = 'qh_climate_crop_growth'
    table_name = table_dict[element]

    # 确定作物· 
    crop_dict = dict()
    crop_dict['spring_maizet'] = '010402'
    crop_dict['spring_wheat'] = '010304'
    crop_dict['winter_wheat'] = '010301'
    crop_dict['highland_barley'] = '010307'
    crop_dict['napus'] = '010603'
    crop_dict['campestris'] = '010602'
    crop_dict['Horsebean'] = '010907'
    crop_dict['potato'] = '010906'
    crop_str = crop_dict[crop]
    
    # 确定要素        
    element_dict = dict()
    element_dict['crop_acreage'] = '待定'
    element_dict['yield'] = '待定'
    element_dict['sowin_date'] = '11'
    element_dict['maturity'] = '91'
    if crop in ['spring_maizet']:
        element_dict['reproductive_period'] = '11,21,31,32,41,61,71,72,73,81,91'
    elif crop in ['spring_wheat','winter_wheat','highland_barley']:
        element_dict['reproductive_period'] = '11,21,22,31,41,51,52,61,62,71,72,81,91'
    elif crop in ['napus','campestris']:
        element_dict['reproductive_period'] = '11,21,31,41,51,61,71,81,82,91'
    elif crop in ['Horsebean']:
        element_dict['reproductive_period'] ='11,21,22,51,71,81,82,91'
    elif crop in ['potato']:
       element_dict['reproductive_period'] ='11,21,51,61,71,91'       
        
    element_dict['reproductive_day'] = '11,91'
    element_str = element_dict[element]
    elements='Station_Id_C,Station_Name,Datetime,Lon,Lat,Year,Mon,Day,crop_name,groper_name_ten'
    
    # 发育期名称
    crop_name_dict=dict()
    if crop in ['spring_maizet']:
        crop_name_dict['reproductive_period'] = '播种,出苗,三叶,移栽,七叶,拔节,抽雄,开花,吐丝,乳熟,成熟'
    elif crop in ['spring_wheat','winter_wheat','highland_barley']:
        crop_name_dict['reproductive_period'] = '播种,出苗,三叶,分蘖,越冬开始,返青,起身,拔节,孕穗,抽穗,开花,乳熟,成熟'
    elif crop in ['napus','campestris']:
        crop_name_dict['reproductive_period'] = '播种,出苗,五真叶,移栽,成活,现蕾,抽薹,开花,绿熟,成熟'
    elif crop in ['Horsebean']:
        crop_name_dict['reproductive_period'] ='播种,出苗,二对真叶,分枝,开花,结荚,鼓粒,成熟'
    elif crop in ['potato']:
       crop_name_dict['reproductive_period'] ='播种,出苗,分支,花絮形成,开花,可收'       
    
    
    # 3. 读取数据
    def get_database_data(elements,element_str,crop_str,sta_ids,table_name,stats_times,station_flag):
        conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
        cur = conn.cursor()
            
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND station_id_c IN %s
                            AND crop_name = %s
                            AND groper_name_ten IN %s
                            
                        """)
    
        # 根据sql获取统计年份data
        sta_ids = tuple(sta_ids.split(','))
        
        if ',' in element_str:
            element_tuple = tuple(element_str.split(','))
        else:
            element_tuple = (element_str)

        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        cur.execute(query, (start_year, end_year, sta_ids,crop_str,element_tuple))
        data = cur.fetchall()
        
        df = pd.DataFrame(data)
        df.columns = elements.split(',')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True, drop=False)
        
        df['date_num'] = df.index.dayofyear
        df['date'] = df['Mon'].astype(str)+'月'+ df['Day'].astype(str)+'日'
        
        
        if station_flag==1:
            station_df=df[['Station_Id_C','Station_Name','Lon','Lat']]
            station_df.drop_duplicates(inplace=True)
            
            return df,station_df
        else:
            return df
        
    data_df_1,station_df= get_database_data(elements,element_str,crop_str,sta_ids,table_name,stats_times,1)
    refer_df = get_database_data(elements,element_str,crop_str,sta_ids,table_name,refer_years,0)
    nearly_df = get_database_data(elements,element_str,crop_str,sta_ids,table_name,nearly_years,0)
    last_df=nearly_df[nearly_df.index.year==last_year].copy()
  
    if crop in ['winter_wheat']:
        data_df_1.index = data_df_1.index - MonthEnd(8)
        refer_df.index = refer_df.index - MonthEnd(8)
        nearly_df.index = nearly_df.index - MonthEnd(8)
        last_df.index = last_df.index - MonthEnd(8)

  
    
    if element in ['sowin_date','maturity']:
        data_date_df = get_database_data(elements,element_str,crop_str,sta_ids,table_name,stats_times,0)
        data_df=data_df_1.pivot_table(index=data_df_1.index, columns=['Station_Id_C'], values='date_num')
        refer_df=refer_df.pivot_table(index=refer_df.index, columns=['Station_Id_C'], values='date_num')
        nearly_df=nearly_df.pivot_table(index=nearly_df.index, columns=['Station_Id_C'], values='date_num')
        last_df=last_df.pivot_table(index=last_df.index, columns=['Station_Id_C'], values='date_num')
        data_date_df=data_df_1.pivot_table(index=data_df_1.index, columns=['Station_Id_C'], values='date',aggfunc='first')

        data_df.index = data_df.index.strftime('%Y')
        refer_df.index = refer_df.index.strftime('%Y')
        nearly_df.index = nearly_df.index.strftime('%Y')
        last_df.index = last_df.index.strftime('%Y')
        data_date_df.index = data_date_df.index.strftime('%Y')

    elif element in ['reproductive_period']:
        
        result=pd.DataFrame()
        for station_id in data_df_1['Station_Name'].unique():
            data_df_2=data_df_1[data_df_1['Station_Name']==station_id]
            data_df_3=data_df_2.pivot_table(index=data_df_2.index, columns=['groper_name_ten'], values='date',aggfunc='first')
            data_df_4= data_df_3.resample('A').first()

            data_df_4.index = data_df_4.index.strftime('%Y')
            data_df_5=pd.DataFrame(index=data_df_4.index,columns=element_dict[element].split(','))   
            
            for columns_id in data_df_5.columns:
                try:
                    data_df_5[columns_id]=data_df_4[columns_id]
                except:
                    continue
        
            data_df_5.columns=crop_name_dict[element].split(',')
            data_df_5.insert(0, '站名', station_id)
            result=pd.concat([result,data_df_5])
        
        return result
    
    elif element in ['reproductive_day']:
        def reproductive_day_deal(crop,df):
            
            df.index = df.index.strftime('%Y')

            df_1=df[df['groper_name_ten']=='91']
            df_2=df[df['groper_name_ten']=='11']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

            df_11=df_1.pivot_table(index=df_1.index, columns=['Station_Id_C'], values='date_num')
            df_21=df_2.pivot_table(index=df_2.index, columns=['Station_Id_C'], values='date_num')

            df_111, df_211 = df_21.align(df_11, join='inner', axis=0)

            if crop in ['winter_wheat']:
                result=365-df_21+df_11
            else:
                result = df_211.sub(df_111, fill_value=0)

            return result
        data_df=reproductive_day_deal(crop,data_df_1)
        refer_df=reproductive_day_deal(crop,refer_df)
        nearly_df=reproductive_day_deal(crop,nearly_df)
        last_df=reproductive_day_deal(crop,last_df)

        
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

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df.apply(trend_rate, axis=0).round(1)
    tmp_df.loc['最大值'] = data_df.iloc[:, :].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[:, :].min(axis=0)
    tmp_df.loc['与上一年比较值'] = (data_df.iloc[:, :].mean(axis=0) - last_df.iloc[:, :].mean(axis=0)).round(1)
    tmp_df.loc['近10年均值'] = nearly_df.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['与近10年比较值'] = (data_df.iloc[:, :].mean(axis=0) - nearly_df.iloc[:, :].mean(axis=0)).round(1)
    tmp_df.loc['参考时段均值'] = refer_df.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']).round(1)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'] / tmp_df.loc['参考时段均值']) * 100).round(2)

    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).round(1)
    stats_result['区域距平'] = (stats_result.iloc[:, :].mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).round(1)
    stats_result['区域距平百分率'] = (stats_result['区域距平']/refer_df.iloc[:, :].mean().mean()).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:, :-3].max(axis=1)
    stats_result['区域最小值'] = stats_result.iloc[:, :-4].min(axis=1)
    
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
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    

    stats_result.reset_index(drop=True, inplace=True)
    post_data_df = data_df.copy()
    post_refer_df = refer_df.copy()
    
    if element in ['sowin_date','maturity']:
        return station_df,stats_result,post_data_df,post_refer_df,reg_params,data_date_df
    else:
        return station_df,stats_result,post_data_df,post_refer_df,reg_params

    
if __name__ == '__main__':
    t1 = time.time()
    data_json = dict()
    data_json['element'] = 'reproductive_period' #sowin_date
    data_json['crop'] = 'winter_wheat'
    data_json['refer_years'] = '2012,2024'
    data_json['nearly_years'] = '2014,2024'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2018,2024'
    data_json['sta_ids'] = '52868,52876'

    result = agriculture_features_stats(data_json)
    t2 = time.time()
    print(t2 - t1)