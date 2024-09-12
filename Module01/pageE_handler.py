# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:12:18 2024

@author: EDY


"""

import os
import uuid
import pandas as pd
import xarray as xr
import psycopg2
from io import StringIO
from psycopg2 import sql
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.data_loader_with_threads import get_cmadaas_yearly_data
from Utils.data_loader_with_threads import get_cmadaas_monthly_data
from Utils.data_loader_with_threads import get_cmadaas_daily_data
from Utils.data_loader_with_threads import get_cmadaas_daily_period_data
from Utils.data_processing import data_processing
from Module01.wrapped.func02_interp_grid import contour_picture
from Module01.wrapped.func03_mk_tests import time_analysis
from Module01.wrapped.func04_cumsum_anomaly import calc_anomaly_cum
from Module01.wrapped.func05_moving_avg import calc_moving_avg
from Module01.wrapped.func06_wavelet_analyse import wavelet_main
from Module01.wrapped.func07_correlation_analysis import correlation_analysis
from Module01.wrapped.func08_eof import eof, reof
from Module01.wrapped.func09_eemd import eemd
import time
from Module01.wrapped.func13_frs_table_stats import frs_table_stats
from Module01.wrapped.func14_ice_table_stats import snow_table_stats

# 冰冻圈

def freeze_features_stats(data_json):
    '''
    获取天擎数据，参数说明
    :param element：对应原型，传入的要素名称
    
       查询统计 - 冰冻圈 - 冻土
       最大冻结深度 FRS_DEPTH
       开始冻结日期 FRS_START
       完全融化日期 FRS_END
       冻结期 FRS_TIME
       
       查询统计 - 冰冻圈 - 积雪
       最大积雪深度：SNOW_DEPTH
       积雪日数： SNOW_DAYS

    :param refer_years: 对应原型的参考时段，只和气候值统计有关，传参：'%Y,%Y'

    :param nearly_years: 传入近10年的年份，以今年为例，传：'1994,2023'

    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'
        季 - 'Q'
        月(连续) - 'M1'
        月(区间) - 'M2' 
        日(连续) - 'D1'
        日(区间) - 'D2'

    :param stats_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'
        (2)当time_freq选择季Q。下载连续的月数据，处理成季数据（多下两个月数据），提取历年相应的季节数据，传：['%Y,%Y','3,4,5']，其中'3,4,5'为季度对应的月份 
        (3)当time_freq选择月(连续)M1。下载连续的月数据，传参：'%Y%m,%Y%m'
        (4)当time_freq选择月(区间)M2。下载连续的月数据，随后提取历年相应的月数据，传参：['%Y,%Y','11,12,1,2'] 前者年份，'11,12,1,2'为连续月份区间
        (5)当time_freq选择日(连续)D1。下载连续的日数据，传参：'%Y%m%d,%Y%m%d'
        (6)当time_freq选择日(区间)D2。直接调天擎接口，下载历年区间时间段内的日数据，传：['%Y,%Y','%m%d,%m%d'] 前者年份，后者区间
    
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
    refer_years = data_json['refer_years']
    nearly_years = data_json['nearly_years']
    time_freq = data_json['time_freq']
    stats_times = data_json['stats_times']
    sta_ids = data_json['sta_ids']
    interp_method = data_json['interp_method']
    ci = data_json['ci']
    shp_path = data_json.get('shp_path')
    degree = data_json.get('degree')
    
    # 2.参数处理
    if shp_path is not None:
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径

    last_year = int(nearly_years.split(',')[-1])  # 上一年的年份
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)
    

    if element in ['FRS_DEPTH','FRS_START','FRS_END','FRS_TIME']:
        element_str = 'frs_1st_top,frs_1st_bot,frs_2nd_top,frs_2nd_bot,frs_state,frs_depth'
        
    elif element in ['SNOW_DEPTH','SNOW_DAYS']:
        element_str = 'snow_depth'


    table_name='qh_climate_cmadaas_day'
    # 从数据库截数据
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
    sta_ids = tuple(sta_ids.split(','))
    elements = 'Station_Id_C,Station_Name,Lon,Lat,Datetime,' + element_str

    if time_freq == 'Y':  # '%Y,%Y'
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND station_id_c IN %s
                        """)

        # 根据sql获取统计年份data
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        cur.execute(query, (start_year, end_year, sta_ids))
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
        
        
    # 统计年份数据处理为df
    data_df = pd.DataFrame(data)
    data_df.columns = elements.split(',')
    
    data_df.set_index('Datetime', inplace=True)
    data_df.index = pd.DatetimeIndex(data_df.index)
    data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)

    if 'Unnamed: 0' in data_df.columns:
        data_df.drop(['Unnamed: 0'], axis=1, inplace=True)   
        
    # 下载参考时段的数据
    query = sql.SQL(f"""
                    SELECT {elements}
                    FROM public.{table_name}
                    WHERE
                        CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                        AND station_id_c IN %s
                    """)

    start_year = refer_years.split(',')[0]
    end_year = refer_years.split(',')[1]
    cur.execute(query, (start_year, end_year, sta_ids))
    data = cur.fetchall()
    refer_df = pd.DataFrame(data)
    refer_df.columns = elements.split(',')
    
    refer_df.set_index('Datetime', inplace=True)
    refer_df.index = pd.DatetimeIndex(refer_df.index)
    refer_df['Station_Id_C'] = refer_df['Station_Id_C'].astype(str)

    if 'Unnamed: 0' in refer_df.columns:
        refer_df.drop(['Unnamed: 0'], axis=1, inplace=True) 
        
    # 下载近10年的数据
    start_year = nearly_years.split(',')[0]
    end_year = nearly_years.split(',')[1]
    cur.execute(query, (start_year, end_year, sta_ids))
    data = cur.fetchall()
    nearly_df = pd.DataFrame(data)
    nearly_df.columns = elements.split(',')
    
    nearly_df.set_index('Datetime', inplace=True)
    nearly_df.index = pd.DatetimeIndex(nearly_df.index)
    nearly_df['Station_Id_C'] = nearly_df['Station_Id_C'].astype(str)

    if 'Unnamed: 0' in nearly_df.columns:
        nearly_df.drop(['Unnamed: 0'], axis=1, inplace=True) 
        
    # 关闭数据库
    cur.close()
    conn.close()

    ###################################################
    # 开始计算
    result_dict = dict()
    result_dict['uuid'] = uuid4

    # 首先获取站号对应的站名
    station_df = pd.DataFrame()
    station_df['站号'] = [
        51886, 51991, 52602, 52633, 52645, 52657, 52707, 52713, 52737, 52745, 52754, 52765, 52818, 52825, 52833, 52836, 52842, 52851, 52853, 52855, 52856, 
        52859, 52862, 52863, 52866, 52868, 52869, 52874, 52875, 52876, 52877, 52908, 52942, 52943, 52955, 52957, 52963, 52968, 52972, 52974, 56004, 56015, 
        56016, 56018, 56021, 56029, 56033, 56034, 56043, 56045, 56046, 56065, 56067, 56125, 56151]
    station_df['站名'] = [
        '茫崖', '那陵格勒', '冷湖', '托勒', '野牛沟', '祁连', '小灶火', '大柴旦', '德令哈', '天峻', '刚察', '门源', '格尔木', '诺木洪', '乌兰', '都兰', '茶卡', 
        '江西沟', '海晏', '湟源', '共和', '瓦里关', '大通', '互助', '西宁', '贵德', '湟中', '乐都', '平安', '民和', '化隆', '五道梁', '河卡', '兴海', '贵南', '同德',
        '尖扎', '泽库', '循化', '同仁', '沱沱河', '曲麻河', '治多', '杂多', '曲麻莱', '玉树', '玛多', '清水河', '玛沁', '甘德', '达日', '河南', '久治', '囊谦', '班玛']
    station_df['站号'] = station_df['站号'].map(str)
    new_station = station_df[ station_df['站号'].isin(sta_ids)]

    # stats_result 展示结果表格
    # post_data_df 统计年份数据，用于后续计算
    # post_refer_df 参考年份数据，用于后续计算

    if element in ['FRS_DEPTH','FRS_TIME']:
        stats_result, post_data_df, post_refer_df, reg_params = frs_table_stats(data_df, refer_df, nearly_df, element,last_year)

    elif element in ['FRS_START','FRS_END']:
        stats_result, post_data_df, post_refer_df, reg_params,data_df_time = frs_table_stats(data_df, refer_df, nearly_df, element,last_year)

    elif element in ['SNOW_DEPTH','SNOW_DAYS']:
        stats_result, post_data_df, post_refer_df, reg_params = snow_table_stats(data_df, refer_df, nearly_df, element,time_freq,last_year)

    # 分布图 try在里面了
    if shp_path is not None:
        nc_path, _, _, _, _ = contour_picture(stats_result, data_df, shp_path, interp_method, data_dir)
        nc_path_trans = nc_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
        nc_path_trans = nc_path_trans.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        print('分布图插值生成nc完成')
    else:
        nc_path = None
        nc_path_trans = None
            
    # 6/7. 统计分析-EOF分析
    if nc_path is not None:
        try:
            ds = xr.open_dataset(nc_path)
            eof_path = eof(ds, shp_path, data_dir)
            reof_path = reof(ds, shp_path, data_dir)
            print('eof/reof完成')
        except:
            eof_path = None
            reof_path = None
            print('没有插值生成网格文件，无法计算eof/reof')

    # 测试下来，只有1个值也能出结果，以下所有的暂时不用加异常处理
    # 1.统计分析-mk检验
    mk_result = time_analysis(post_data_df, data_dir)
    print('MK检验完成')

    # 2.统计分析-累积距平
    anomaly_result = calc_anomaly_cum(post_data_df, post_refer_df, data_dir)
    print('距平完成')

    # 3.统计分析-滑动平均
    moving_result = calc_moving_avg(post_data_df, 5, data_dir)
    print('滑动平均完成')

    # 4. 统计分析-小波分析
    wave_result = wavelet_main(post_data_df, data_dir)
    print('小波完成')

    # 5. 统计分析-相关分析
    correlation_result = correlation_analysis(post_data_df, data_dir)
    print('相关分析完成')

    # 8.EEMD分析
    eemd_result = eemd(post_data_df, data_dir)
    print('eemd完成')

    # 数据保存
    result_dict['表格'] = dict()
    result_dict['表格'] = stats_result.to_dict(orient='records')

    if element in ['FRS_START','FRS_END']:
        result_dict['表格_日期'] =data_df_time

    result_dict['分布图'] = dict()
    result_dict['分布图'] = nc_path_trans
    result_dict['统计分析'] = dict()
    result_dict['统计分析']['线性回归'] = reg_params.to_dict(orient='records')
    result_dict['统计分析']['MK检验'] = mk_result
    result_dict['统计分析']['累积距平'] = anomaly_result
    result_dict['统计分析']['滑动平均'] = moving_result
    result_dict['统计分析']['小波分析'] = wave_result
    result_dict['统计分析']['相关分析'] = correlation_result
    result_dict['统计分析']['EOF分析'] = eof_path
    result_dict['统计分析']['REOF分析'] = reof_path
    result_dict['统计分析']['EEMD分析'] = eemd_result
    result_dict['站号'] = new_station.to_dict(orient='records')

    return result_dict


if __name__ == '__main__':
    t1 = time.time()
    data_json = dict()
    data_json['element'] = 'FRS_DEPTH'
    data_json['refer_years'] = '2021,2022'
    data_json['nearly_years'] = '2021,2022'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2021,2024'  # '198105,202009' # '1981,2023'
    data_json['sta_ids'] = '51886,52737,52842,52886,52876'
    data_json['interp_method'] = 'idw'
    data_json['ci'] = 95
    data_json['shp_path'] = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\01-青海省\青海省县级数据.shp'
    data_json['degree'] = 10
    
    result = freeze_features_stats(data_json)
    t2 = time.time()
    print(t2 - t1)
