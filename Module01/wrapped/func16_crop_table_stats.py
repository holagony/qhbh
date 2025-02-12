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

        农作物：1 crop
        粮食作物：2 food_crop
        
    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'
        有效时间： 2010 - 2024
    :param stats_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'

    :param sta_ids: 传入的站点，多站，传：'52866,52713,52714'
                    传入的区域，全省：630000；各区域id
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
    crop = data_json.get('crop','spring_maizet')
    refer_years = data_json['refer_years']
    nearly_years = data_json['nearly_years']
    stats_times = data_json['stats_times']
    sta_ids = data_json['sta_ids']
    time_freq = data_json['time_freq']
    #last_year = int(nearly_years.split(',')[-1])  # 上一年的年份

    # 2.参数处理
    # 确定表名
    table_dict = dict()
    table_dict['crop_acreage'] = 'qh_climate_crop_sowing_area'
    table_dict['yield'] = '待定'
    table_dict['sowin_date'] = 'qh_climate_crop_growth'
    table_dict['maturity'] = 'qh_climate_crop_growth'
    table_dict['reproductive_period'] = 'qh_climate_crop_growth'
    table_dict['reproductive_day'] = 'qh_climate_crop_growth'
    
    if element=='crop_acreage':
        if sta_ids=='630000':
            table_name='qh_climate_crop_sowing_province'
        else:
            table_name='qh_climate_crop_sowing_area'

    elif element=='yield':
        if sta_ids=='630000':
            table_name='qh_climate_crop_yield_province'
        else:
            table_name='qh_climate_crop_yield'    
    else:
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
    crop_dict['crop'] = 1
    crop_dict['food_crop'] = 2

    crop_str = crop_dict[crop]
    
    
    # 确定要素        
    element_dict = dict()
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

    if element=='crop_acreage':
        if sta_ids=='630000':
            elements='datatime,total,grain,wheat,tubers,cashcrop'
        else:
            elements='datatime,zone,lon,lat,sowing_area,type,station_id_c'

    elif element=='yield':
        if sta_ids=='630000':
            elements='datatime,grain,wheat,tubers,oilseeds'
        else:
            elements='datatime,zone,lon,lat,yield,type,station_id_c'    
    else:
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
    
    # 区域名
    zone=['城东区', '城中区', '城西区', '城北区', '西宁市', '大通县', '湟中县', '湟源县', '平安县', '民和县', '乐都县', '互助县', '化隆县', '循化县', '门源县', '祁连县', '海晏县', '刚察县', '共和县', '贵德县', '贵南县', '同德县', '兴海县', '同仁县', '尖扎县', '泽库县', '河南县', '玛沁县', '班玛县', '甘德县', '达日县', '久治县', '玛多县', '玉树县', '杂多县', '称多县', '治多县', '囊谦县', '曲麻莱县', '格尔木市', '德令哈镇', '乌兰县', '都兰县', '天峻县', '茫崖行委', '大柴旦镇', '冷湖镇']

    # 3. 读取数据
    def get_database_data(elements,element_str,crop_str,sta_ids,table_name,stats_times,station_flag,time_freq):
        '''
        从数据库获取数据
        '''
        sta_ids = tuple(sta_ids.split(','))
        
        if ',' in element_str:
            element_tuple = tuple(element_str.split(','))
        else:
            element_tuple = (element_str,)
                
        conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
        cur = conn.cursor()
        
        if time_freq == 'Y':  # '%Y,%Y'
                
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
            start_year = stats_times.split(',')[0]
            end_year = stats_times.split(',')[1]
            cur.execute(query, (start_year, end_year, sta_ids,crop_str,element_tuple))
            data = cur.fetchall()
            
    
        elif time_freq == 'Q':  # ['%Y,%Y','3,4,5']
            mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
            mon_ = tuple(mon_list)
            years = stats_times[0]
            start_year = years.split(',')[0]
            end_year = years.split(',')[1]
    
            if 12 in mon_list:
                query = sql.SQL(f"""
                                SELECT {elements}
                                FROM public.{table_name}
                                    WHERE (SUBSTRING(datetime, 1, 4) BETWEEN %s AND %s) 
                                    AND SUBSTRING(datetime, 6, 2) IN ('12', '01', '02')
                                    OR (SUBSTRING(datetime, 1, 4) = %s AND SUBSTRING(datetime, 6, 2) = '12')
                                    OR (SUBSTRING(datetime, 1, 4) = %s AND SUBSTRING(datetime, 6, 2) IN ('01', '02'))
                                    AND station_id_c IN %s
                                    AND crop_name = %s
                                    AND groper_name_ten IN %s
                                """)
                cur.execute(query, (start_year, end_year,str(int(start_year)-1),str(int(end_year)+1), sta_ids))
    
            else:    
                query = sql.SQL(f"""
                                SELECT {elements}
                                FROM public.{table_name}
                                WHERE
                                    (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                    AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                                    AND station_id_c IN %s
                                    AND crop_name = %s
                                    AND groper_name_ten IN %s
                                """)  
                cur.execute(query, (start_year, end_year, mon_, sta_ids))
    
            data = cur.fetchall()
    
        elif time_freq == 'M1':  # '%Y%m,%Y%m'
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                            WHERE
                                ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s))
                                AND station_id_c IN %s
                                AND crop_name = %s
                                AND groper_name_ten IN %s
                            """)
    
            start_year = stats_times.split(',')[0][:4]
            end_year = stats_times.split(',')[1][:4]
            start_month = stats_times.split(',')[0][4:]
            end_month = stats_times.split(',')[1][4:]
            cur.execute(query, (start_year, start_month, start_year, end_year, end_year, end_month, sta_ids))
            data = cur.fetchall()
    
        elif time_freq == 'M2':  # ['%Y,%Y','11,12,1,2']
            mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
            mon_ = tuple(mon_list)
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                            WHERE
                                (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                                AND station_id_c IN %s
                                AND crop_name = %s
                                AND groper_name_ten IN %s
                            """)
    
            years = stats_times[0]
            start_year = years.split(',')[0]
            end_year = years.split(',')[1]
            cur.execute(query, (start_year, end_year, mon_, sta_ids))
            data = cur.fetchall()
    
        elif time_freq == 'D1':  # '%Y%m%d,%Y%m%d'
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                            WHERE
                                ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s))
                                AND station_id_c IN %s
                                AND crop_name = %s
                                AND groper_name_ten IN %s
                            """)
    
            start_year = stats_times.split(',')[0][:4]
            end_year = stats_times.split(',')[1][:4]
            start_month = stats_times.split(',')[0][4:6]
            end_month = stats_times.split(',')[1][4:6]
            start_date = stats_times.split(',')[0][6:]
            end_date = stats_times.split(',')[1][6:]
            cur.execute(query, (start_year, start_month, start_date, start_year, end_year, end_year, end_month, end_date, sta_ids))
            data = cur.fetchall()
    
        elif time_freq == 'D2':  # ['%Y,%Y','%m%d,%m%d']
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                            WHERE
                                (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND (
                                    (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                                    OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) < %s)
                                    OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s)
                                ))
                                AND station_id_c IN %s
                                AND crop_name = %s
                                AND groper_name_ten IN %s
                            """)
    
            years = stats_times[0]
            dates = stats_times[1]
            start_year = years.split(',')[0]
            end_year = years.split(',')[1]
            start_mon = dates.split(',')[0][:2]
            end_mon = dates.split(',')[1][:2]
            start_date = dates.split(',')[0][2:]
            end_date = dates.split(',')[1][2:]
            cur.execute(query, (start_year, end_year, start_mon, start_date, start_mon, end_mon, end_mon, end_date, sta_ids))
            data = cur.fetchall()
    
        cur.close()
        conn.close()
        try:
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
        
        except Exception:
            raise Exception('数据库选择的时段无数据')
    
    
    # 播种、成熟、发育读取数据库的方法
    if element in ['sowin_date','maturity','reproductive_period','reproductive_day']:
        data_df_1,station_df= get_database_data(elements,element_str,crop_str,sta_ids,table_name,stats_times,1,time_freq)
        refer_df = get_database_data(elements,element_str,crop_str,sta_ids,table_name,refer_years,0,time_freq)
        nearly_df = get_database_data(elements,element_str,crop_str,sta_ids,table_name,nearly_years,0,time_freq)
        last_df=nearly_df[nearly_df.index.year==nearly_df.index.year[-1]].copy()
   
    # 解决冬小麦 跨年问题
    if crop in ['winter_wheat']:
        data_df_1.index = data_df_1.index - MonthEnd(8)
        refer_df.index = refer_df.index - MonthEnd(8)
        nearly_df.index = nearly_df.index - MonthEnd(8)
        last_df.index = last_df.index - MonthEnd(8)

     # 播种、成熟的表格
    if element in ['sowin_date','maturity']:
        data_date_df = get_database_data(elements,element_str,crop_str,sta_ids,table_name,stats_times,0,time_freq)
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

    # 生育期表格
    elif element in ['reproductive_period']:
        
        result=pd.DataFrame()
        result2=pd.DataFrame()
        for station_id in data_df_1['Station_Name'].unique():
            
            data_df_2=data_df_1[data_df_1['Station_Name']==station_id]
            data_df_3=data_df_2.pivot_table(index=data_df_2.index, columns=['groper_name_ten'], values='date',aggfunc='first')
            data_df_4= data_df_3.resample('A').first()
            data_df_4.index = data_df_4.index.strftime('%Y')
            
            data_df_42=pd.DataFrame(index=data_df_4.index,columns=data_df_4.columns)   
            for col in data_df_4.columns:
                date_strings = data_df_4.index.astype(str) + '年' + data_df_4[col].astype(str)
                dates = pd.to_datetime(date_strings, format='%Y年%m月%d日', errors='coerce')

                day_of_year = dates.dt.dayofyear
                data_df_42[col] = day_of_year
            data_df_42 = data_df_42.replace({np.nan: None})
            
            data_df_5=pd.DataFrame(index=data_df_4.index,columns=element_dict[element].split(','))   
            data_df_6=pd.DataFrame(index=data_df_4.index,columns=element_dict[element].split(','))   

            for columns_id in data_df_5.columns:
                try:
                    data_df_5[columns_id]=data_df_4[columns_id]
                    data_df_6[columns_id]=data_df_42[columns_id]
                    
                except:
                    continue
        
            data_df_5.columns=crop_name_dict[element].split(',')
            data_df_6.columns=crop_name_dict[element].split(',')
            station_name = station_id[:station_id.find('国')] if '国' in station_id else station_id
            data_df_5.insert(0, '站名', station_name)
            data_df_6.insert(0, '站名', station_name)
            result=pd.concat([result,data_df_5])
            result2=pd.concat([result2,data_df_6])
        return result,result2
   
    # 生育期天数
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

    # 产量和面积
    elif element in ['crop_acreage','yield']:
        if sta_ids=='630000':
            def acreage_yield_province_get(element,elements,table_name,stats_times):
                conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
                cur = conn.cursor()
                    
                query = sql.SQL(f"""
                                SELECT {elements}
                                FROM public.{table_name}
                                WHERE
                                    CAST(SUBSTRING(datatime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s                                
                                """)
                    
                start_year = stats_times.split(',')[0]
                end_year = stats_times.split(',')[1]
                cur.execute(query, (start_year, end_year))
                data = cur.fetchall()
                
                df = pd.DataFrame(data)
                df.columns = elements.split(',')
                df['datatime'] = pd.to_datetime(df['datatime'])
                df.set_index('datatime', inplace=True, drop=True)
                df.index = df.index.strftime('%Y')
                
                df=df.astype(float)
                if element=='crop_acreage':
                    df.columns=['总播种面积','粮食','小麦','薯类','经济作物']
                elif element=='yield':
                    df.columns=['粮食','小麦','薯类','油料']
                
                df=df.astype(float).round(1)
                return df

            data_df= acreage_yield_province_get(element,elements,table_name,stats_times)
            refer_df = acreage_yield_province_get(element,elements,table_name,refer_years)
            nearly_df = acreage_yield_province_get(element,elements,table_name,nearly_years)
            last_df=nearly_df[nearly_df.index==nearly_df.index[-1]].copy()
            station_df=pd.DataFrame(columns=['站点','站名','经度','纬度 '])
        else:
            def acreage_yield_get(elements,table_name,stats_times,sta_ids,types,station_flag):
                conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
                cur = conn.cursor()
                    
                query = sql.SQL(f"""
                                SELECT {elements}
                                FROM public.{table_name}
                                WHERE
                                    CAST(SUBSTRING(datatime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                    AND station_id_c IN %s
                                    AND type = %s
                                    
                                """)
                sta_ids = tuple(sta_ids.split(','))
                start_year = stats_times.split(',')[0]
                end_year = stats_times.split(',')[1]
                cur.execute(query, (start_year, end_year, sta_ids,types))
                data = cur.fetchall()
                
                df = pd.DataFrame(data)
                df.columns = elements.split(',')
                df['datatime'] = pd.to_datetime(df['datatime'])
                df.set_index('datatime', inplace=True, drop=True)
                df.index = df.index.strftime('%Y')
                
                station_df=df[['station_id_c','zone','lon','lat']]
                station_df.drop_duplicates(inplace=True)
                
                if element=='crop_acreage':
                    df=df.pivot_table(index=df.index, columns=['station_id_c'], values='sowing_area')
                else:
                    df=df.pivot_table(index=df.index, columns=['station_id_c'], values='yield')

                df=df.astype(float).round(1)
                
                station_df.reset_index(inplace=True,drop=True)
                existing_columns = [col for col in zone if col in station_df['zone'].values]
                existing_df = pd.DataFrame({'zone': existing_columns})
                station_df = station_df.merge(existing_df, on='zone', how='right')
                

                df=df.astype(float).round(1)
                    
                if station_flag==1:

                    return df,station_df
                
                else:

                    return df

            data_df,station_df= acreage_yield_get(elements,table_name,stats_times,sta_ids,crop_str,1)
            refer_df = acreage_yield_get(elements,table_name,refer_years,sta_ids,crop_str,0)
            nearly_df = acreage_yield_get(elements,table_name,nearly_years,sta_ids,crop_str,0)
            last_df=nearly_df[nearly_df.index==nearly_df.index[-1]].copy()
            
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
    
    if element not in ['crop_acreage','yield']:
        stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).round(1)
        stats_result['区域距平'] = (stats_result.iloc[:, :].astype(float).mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).round(1)
        stats_result['区域距平百分率'] = (stats_result['区域距平']/refer_df.iloc[:, :].mean().mean()).round(1)
        stats_result['区域最大值'] = stats_result.iloc[:, :-3].max(axis=1)
        stats_result['区域最小值'] = stats_result.iloc[:, :-4].min(axis=1)
    
    # 在concat前增加回归方程
    def lr(x):
        try:
            x = x.to_frame()
            x['num'] = np.arange(len(x))+1

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
    data_json['element'] = 'crop_acreage' #sowin_date
    data_json['crop'] = 'crop'
    data_json['refer_years'] = '2002,2024'
    data_json['nearly_years'] = '2004,2024'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2008,2024'
    data_json['sta_ids'] = "52866,52866,52866,52866,52866,52862,52869,52855"

    station_df,stats_result,post_data_df,post_refer_df,reg_params = agriculture_features_stats(data_json)
    t2 = time.time()
    print(t2 - t1)