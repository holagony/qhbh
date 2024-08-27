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
from Module01.wrapped.func10_other_table_stats import other_table_stats
from Module01.wrapped.func11_pre_table_stats import pre_table_stats
from Module01.wrapped.func12_tem_table_stats import tem_table_stats

from Module01.wrapped.func02_interp_grid import contour_picture
from Module01.wrapped.func03_mk_tests import time_analysis
from Module01.wrapped.func04_cumsum_anomaly import calc_anomaly_cum
from Module01.wrapped.func05_moving_avg import calc_moving_avg
from Module01.wrapped.func06_wavelet_analyse import wavelet_main
from Module01.wrapped.func07_correlation_analysis import correlation_analysis
from Module01.wrapped.func08_eof import eof, reof
from Module01.wrapped.func09_eemd import eemd


def extreme_climate_features(data_json):
    '''
    获取天擎数据，参数说明
    :param element：对应原型，传入的要素名称
    极端气温指数：
        冷夜日数  TN10p  特别参数：l_data =
        冷昼日数  TX10p  特别参数：l_data =
        暖夜日数  TN90p  特别参数：n_data =
        暖昼日数  TX90p  特别参数：n_data =
        冰封日数  ID
        霜冻日数  FD
        最高气温极大值  TXx
        最高气温极小值 TXn
        最低气温极大值  TNx
        最低气温极小值  TNn
        暖持续指数  WSDI
        冷持续指数  CSDI
        气温日较差  DTR
        生长期长度 GSL
        夏季日数  SU
        热夜日数  TR
    极端降水指数：
        持续干期  CDD
        持续湿期  CWD
        降水总量  RZ
        降水日数  RZD
        降水强度  SDII
        强降水量  R95
        强降水日数  R95%D
        特强降水  R50
        特强降水日数  R50D
        中雨日数  R10D
        大雨日数  R25D
        1日最大降水  Rx1day
        5日最大降水  Rx5day
        自定义：
            降雨量： R  特别参数：R  R_flag
            降雨日：  RD 特别参数：RD  RD_flag
            N日降水量：Rxxday  特别参数：Rxxday
    其他气候指数：
        冰雹日数  Hail Hail_Days Hail_Days
        大风日数  GaWIN  GaWIN_Days GaWIN_Days
        沙尘日数： 
            沙尘日数：sa	sa	sa
            沙尘暴日数：SaSt	SaSt_Days	SaSt_Days
            扬沙日数： FlSa	FlSa_Days	FlSa_Days
            浮尘日数：FlDu	FlDu_Days	FlDu_Days
        暴雨日数  rainstorm	rainstorm	rainstorm
        雪灾日数：
            雪灾日数  snow	snow	snow
            轻度雪灾日数  light_snow	light_snow	light_snow
            中度雪灾日数  medium_snow	medium_snow	medium_snow
            重度雪灾日数  heavy_snow	heavy_snow	heavy_snow
            特重度雪灾日数  severe_snow	severe_snow	severe_snow           
        高温日数  high_tem	high_tem	high_tem

        雷暴日数 Thund	Thund_Days	Thund_Days
        气象干旱日数：
            气象干旱日数：drought	drought	drought
            轻度干旱日数：light_drought	light_drought	light_drought
            中度干旱日数：medium_drought	medium_drought	medium_drought
            重度干旱日数：heavy_drought	heavy_drought	heavy_drought
            特重度干旱日数：severe_drought	severe_drought	severe_drought

        
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
    shp_path = data_json['shp_path']

    l_data = data_json.get('l_data')
    n_data = data_json.get('n_data')
    R = data_json.get('R')
    R_flag = data_json.get('R_flag')
    RD = data_json.get('RD')
    RD_flag = data_json.get('RD_flag')
    Rxxday = data_json.get('Rxxday')


    # 2.参数处理
    last_year = int(nearly_years.split(',')[-1])  # 上一年的年份
    uuid4 = uuid.uuid4().hex

    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)
        
    shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径

    # 3. 要素字典
    tem_table=['TN10p','TX10p','TN90p','TX90p','ID','FD','TNx','TXx','TNn','TXn',
               'DTR','WSDI','CSDI','SU','TR','GSL']
    
    pre_table=['CDD','CWD','RZ','RZD','SDII','R95','R95','R50','R50D','R10D',
               'R25D','Rx1day','Rx5day','R','RD','Rxxday']
    
    other_table=['Hail','GaWIN','Thund','SaSt','FlSa','FlDu','sa','rainstorm',
                 'light_snow','snow','medium_snow','heavy_snow','severe_snow',
                  'high_tem','drought','light_drought','medium_drought','heavy_drought',
                  'severe_drought']
    
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
    
    # 极端降水指数
    ele_dict['CDD']='PRE_Time_2020'
    ele_dict['CWD']='PRE_Time_2020'
    ele_dict['RZ']='PRE_Time_2020'
    ele_dict['RZD']='PRE_Time_2020'
    ele_dict['SDII']='PRE_Time_2020'
    ele_dict['R95']='PRE_Time_2020'
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
    ele_dict['Hail ']='Hail_Days'
    ele_dict['GaWIN']='GaWIN_Days'
    ele_dict['sa']='sa'
    ele_dict['SaSt']='SaSt_Days'
    ele_dict['FlSa']='FlSa_Days'
    ele_dict['FlDu']='FlDu_Days'
    ele_dict['rainstorm']='rainstorm'
    ele_dict['snow']='snow'
    ele_dict['light_snow']='light_snow'
    ele_dict['medium_snow']='medium_snow'
    ele_dict['heavy_snow']='heavy_snow'
    ele_dict['severe_snow']='severe_snow'
    ele_dict['high_tem']='high_tem'
    ele_dict['Thund']='Thund_Days'
    ele_dict['drought']='drought'
    ele_dict['light_drought']='light_drought'
    ele_dict['medium_drought']='medium_drought'
    ele_dict['heavy_drought']='heavy_drought'
    ele_dict['severe_drought']='severe_drought'

    ele_day=['Hail','GaWIN','Thund','SaSt','FlSa','FlDu']
    ele_caculate=['sa','rainstorm','light_snow','snow','medium_snow','heavy_snow','severe_snow',
                  'high_tem','drought','light_drought','medium_drought','heavy_drought','severe_drought']
    
    sql_dict=dict()
    sql_dict['day']='qh_qhbh_cmadaas_day'
    sql_dict['day_cal']='qh _qhbh_calc_elements_day'
    sql_dict['mon']='qh_qhbh_cmadaas_month'
    sql_dict['mon_cal']='qh_qhbh_calc_elements_month'
    sql_dict['year']='qh_qhbh_cmadaas_year'
    sql_dict['year_cal']='qh_qhbh_calc_elements_year'

    if not cfg.INFO.READ_LOCAL:

        # 3.解析要下载的参数
        ele = ''
        elements_list = ['TEM_Avg', 'TEM_Max', 'TEM_Min', 'PRE_Time_2020', 'PRE_Days', 'PRE_Max_Day', 'PRS_Avg', 'PRS_Max', 'PRS_Min', 
                         'WIN_S_2mi_Avg', 'WIN_S_Max', 'WIN_S_Inst_Max', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Max', 'GST_Min', 'GST_Avg_5cm', 'GST_Avg_10cm',
                         'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 
                         'SSH', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg', 'RHU_Min']
        resample_max = ['TEM_Max', 'PRS_Max', 'WIN_S_Max', 'WIN_S_Inst_Max', 'GST_Max']
        resample_min = ['TEM_Min', 'PRS_Min', 'GST_Min', 'RHU_Min']
        resample_sum = ['PRE_Time_2020', 'PRE_Days']
        resample_mean = ['TEM_Avg', 'PRS_Avg', 'WIN_S_2mi_Avg', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Avg_5cm', 'GST_Avg_10cm', 'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 
                         'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 'SSH', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg']

        if element in elements_list:
            ele += element

        elif element == 'WIND':
            if time_freq in ['D1', 'D2']:
                ele += 'WIN_D_S_Max'
            else:
                ele += 'WIN_D_S_Max_C'

        elif element == 'EVP_Taka':
            ele += 'PRE_Time_2020,TEM_Avg'

        elif element == 'EVP_Penman':
            ele += 'PRS_Avg,WIN_S_2mi_Avg,TEM_Max,TEM_Min,RHU_Avg,SSH'

        # 4.下载 and 后处理
        if time_freq == 'Y':
            # 下载统计年份的数据
            years = stats_times
            data_df = get_cmadaas_yearly_data(years, ele, sta_ids)
            data_df = data_processing(data_df, element)

            # 下载参考时段的数据
            refer_df = get_cmadaas_yearly_data(refer_years, ele, sta_ids)
            refer_df = data_processing(refer_df, element)

            # 下载近10年的数据
            nearly_df = get_cmadaas_yearly_data(nearly_years, ele, sta_ids)
            nearly_df = data_processing(nearly_df, element)

        elif time_freq == 'Q':
            # 下载统计年份的数据
            mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
            years = stats_times[0]
            mon = '01,12'
            data_df = get_cmadaas_monthly_data(years, mon, ele, sta_ids)
        
            # TODO if element in ['EVP_Penman', 'EVP_taka']:
            data_df = data_df[data_df['Mon'].isin(mon_list)]
            data_df = data_processing(data_df, element)

            # 下载参考时段的数据
            refer_df = get_cmadaas_monthly_data(refer_years, mon, ele, sta_ids)
            refer_df = refer_df[refer_df['Mon'].isin(mon_list)]
            refer_df = data_processing(refer_df, element)

            # 下载近10年的数据
            nearly_df = get_cmadaas_monthly_data(nearly_years, mon, ele, sta_ids)
            nearly_df = nearly_df[nearly_df['Mon'].isin(mon_list)]
            nearly_df = data_processing(nearly_df, element)

        elif time_freq == 'M1':
            # 下载统计年份的数据
            start_time = stats_times.split(',')[0]
            end_time = stats_times.split(',')[1]
            years = start_time[:4] + ',' + end_time[:4]
            mon = start_time[4:] + ',' + end_time[4:]
            data_df = get_cmadaas_monthly_data(years, mon, ele, sta_ids)
            data_df = data_processing(data_df, element)

            # 下载参考时段的数据
            refer_df = get_cmadaas_monthly_data(refer_years, mon, ele, sta_ids)
            refer_df = data_processing(refer_df, element)

            # 下载近10年的数据
            nearly_df = get_cmadaas_monthly_data(nearly_years, mon, ele, sta_ids)
            nearly_df = data_processing(nearly_df, element)

        elif time_freq == 'M2':
            # 下载统计年份的数据
            mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]
            years = stats_times[0]
            mon = '01,12'
            data_df = get_cmadaas_monthly_data(years, mon, ele, sta_ids)
            data_df = data_df[data_df['Mon'].isin(mon_list)]  # 按区间提取月份
            data_df = data_processing(data_df, element)

            # 下载参考时段的数据
            refer_df = get_cmadaas_monthly_data(refer_years, mon, ele, sta_ids)
            refer_df = refer_df[refer_df['Mon'].isin(mon_list)]
            refer_df = data_processing(refer_df, element)

            # 下载近10年的数据
            nearly_df = get_cmadaas_monthly_data(nearly_years, mon, ele, sta_ids)
            nearly_df = nearly_df[nearly_df['Mon'].isin(mon_list)]
            nearly_df = data_processing(nearly_df, element)

        elif time_freq == 'D1':
            # 下载统计年份的数据
            start_time = stats_times.split(',')[0]
            end_time = stats_times.split(',')[1]
            years = start_time[:4] + ',' + end_time[:4]
            date = start_time[4:] + ',' + end_time[4:]
            data_df = get_cmadaas_daily_data(years, date, ele, sta_ids)
            data_df = data_processing(data_df, element)

            # 下载参考时段的数据
            refer_df = get_cmadaas_daily_data(refer_years, date, ele, sta_ids)
            refer_df = data_processing(refer_df, element)

            # 下载近10年的数据
            nearly_df = get_cmadaas_daily_data(nearly_years, date, ele, sta_ids)
            nearly_df = data_processing(nearly_df, element)

        elif time_freq == 'D2':
            # 下载统计年份的数据
            years = stats_times[0]
            date = stats_times[1]
            data_df = get_cmadaas_daily_period_data(years, date, ele, sta_ids)
            data_df = data_processing(data_df, element)

            # 下载参考时段的数据
            refer_df = get_cmadaas_daily_period_data(refer_years, date, ele, sta_ids)
            refer_df = data_processing(refer_df, element)

            # 下载近10年的数据
            nearly_df = get_cmadaas_daily_period_data(nearly_years, date, ele, sta_ids)
            nearly_df = data_processing(nearly_df, element)

    # 走数据库
    else:
        conn = psycopg2.connect(database=cfg.INFO.DB_NAME, 
                                user=cfg.INFO.DB_USER, 
                                password=cfg.INFO.DB_PWD, 
                                host=cfg.INFO.DB_HOST, 
                                port=cfg.INFO.DB_PORT)
        cur = conn.cursor()
        
        if time_freq == 'Y':
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            sta_ids = tuple(sta_ids.split(','))
            
            if element in ele_caculate:
                sql_choose=sql_dict['year_cal']
            else:
                sql_choose=sql_dict['year']
                
            if element not in other_table:
                sql_choose=sql_dict['day']

            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
                            WHERE
                                CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND station_id_c IN %s
                            """)

            # 下载统计年份的数据
            start_year = stats_times.split(',')[0]
            end_year = stats_times.split(',')[1]
            cur.execute(query, (start_year, end_year, sta_ids))
            data = cur.fetchall()
            data_df = pd.DataFrame(data)
            data_df.columns = elements.split(',')
            
            if element in other_table:
                data_df = data_processing(data_df, ele_dict[element])
            else:
                data_df.set_index('Datetime', inplace=True)
                data_df.index = pd.DatetimeIndex(data_df.index)
                data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in data_df.columns:
                    data_df.drop(['Unnamed: 0'], axis=1, inplace=True)

            # 下载参考时段的数据
            start_year = refer_years.split(',')[0]
            end_year = refer_years.split(',')[1]
            cur.execute(query, (start_year, end_year, sta_ids))
            data = cur.fetchall()
            refer_df = pd.DataFrame(data)
            refer_df.columns = elements.split(',')
            
            if element in other_table:
                refer_df = data_processing(refer_df, ele_dict[element])
            else:
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
                                  
            if element in other_table:
                nearly_df = data_processing(nearly_df, ele_dict[element])
            else:
                nearly_df.set_index('Datetime', inplace=True)
                nearly_df.index = pd.DatetimeIndex(nearly_df.index)
                nearly_df['Station_Id_C'] = nearly_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in nearly_df.columns:
                    nearly_df.drop(['Unnamed: 0'], axis=1, inplace=True)
                    
                    

        elif time_freq == 'Q': # ['%Y,%Y','3,4,5']
        
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]


            if element in ele_caculate:
                sql_choose=sql_dict['mon_cal']
            else:
                sql_choose=sql_dict['mon']
                
            if element not in other_table:
                sql_choose=sql_dict['day']

            sta_ids = tuple(sta_ids.split(','))
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
            mon_ = tuple(mon_list)
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
                            WHERE
                                (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                                AND station_id_c IN %s
                            """)

            # 下载统计年份的数据
            years = stats_times[0]
            start_year = years.split(',')[0]
            end_year = years.split(',')[1]
            cur.execute(query, (start_year, end_year, mon_, sta_ids))
            data = cur.fetchall()
            data_df = pd.DataFrame(data)
            data_df.columns = elements.split(',')
            if element in other_table:
                data_df = data_processing(data_df, ele_dict[element])
            else:
                data_df.set_index('Datetime', inplace=True)
                data_df.index = pd.DatetimeIndex(data_df.index)
                data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in data_df.columns:
                    data_df.drop(['Unnamed: 0'], axis=1, inplace=True)            
            
            # 下载参考时段的数据
            if element in ele_caculate:
                sql_choose=sql_dict['year_cal']
            else:
                sql_choose=sql_dict['year']
                
            if element not in other_table:
                sql_choose=sql_dict['day']

            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
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
            
            if element in other_table:
                refer_df = data_processing(refer_df, ele_dict[element])
            else:
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
                      
            if element in other_table:
                nearly_df = data_processing(nearly_df, ele_dict[element])
            else:
                nearly_df.set_index('Datetime', inplace=True)
                nearly_df.index = pd.DatetimeIndex(nearly_df.index)
                nearly_df['Station_Id_C'] = nearly_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in nearly_df.columns:
                    nearly_df.drop(['Unnamed: 0'], axis=1, inplace=True)
                    
        elif time_freq == 'M1': # '%Y%m,%Y%m'
        
            if element in ele_caculate:
                sql_choose=sql_dict['mon_cal']
            else:
                sql_choose=sql_dict['mon']
                
            if element not in other_table:
                sql_choose=sql_dict['day']
                
            sta_ids = tuple(sta_ids.split(','))
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
                            WHERE
                                ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s))
                                AND station_id_c IN %s
                            """)

            # 下载统计年份的数据
            start_year = stats_times.split(',')[0][:4]
            end_year = stats_times.split(',')[1][:4]
            start_month = stats_times.split(',')[0][4:]
            end_month = stats_times.split(',')[1][4:]
            cur.execute(query, (start_year, start_month, start_year, end_year, end_year, end_month, sta_ids))
            data = cur.fetchall()
            data_df = pd.DataFrame(data)
            data_df.columns = elements.split(',')
            
            if element in other_table:
                data_df = data_processing(data_df, ele_dict[element])
            else:
                data_df.set_index('Datetime', inplace=True)
                data_df.index = pd.DatetimeIndex(data_df.index)
                data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in data_df.columns:
                    data_df.drop(['Unnamed: 0'], axis=1, inplace=True)
                    
            # 下载参考时段的数据
            if element in ele_caculate:
                sql_choose=sql_dict['year_cal']
            else:
                sql_choose=sql_dict['year']
                
            if element not in other_table:
                sql_choose=sql_dict['day']

            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
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
            
            if element in other_table:
                refer_df = data_processing(refer_df, ele_dict[element])
            else:
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
                      
            if element in other_table:
                nearly_df = data_processing(nearly_df, ele_dict[element])
            else:
                nearly_df.set_index('Datetime', inplace=True)
                nearly_df.index = pd.DatetimeIndex(nearly_df.index)
                nearly_df['Station_Id_C'] = nearly_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in nearly_df.columns:
                    nearly_df.drop(['Unnamed: 0'], axis=1, inplace=True)
                    
        elif time_freq == 'M2': # ['%Y,%Y','11,12,1,2']
        
            if element in ele_caculate:
                sql_choose=sql_dict['mon_cal']
            else:
                sql_choose=sql_dict['mon']
                
            if element not in other_table:
                sql_choose=sql_dict['day']
                
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            sta_ids = tuple(sta_ids.split(','))
            mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
            mon_ = tuple(mon_list)
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
                            WHERE
                                (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                                AND station_id_c IN %s
                            """)

            # 下载统计年份的数据
            years = stats_times[0]
            start_year = years.split(',')[0]
            end_year = years.split(',')[1]
            cur.execute(query, (start_year, end_year, mon_, sta_ids))
            data = cur.fetchall()
            data_df = pd.DataFrame(data)
            data_df.columns = elements.split(',')
            if element not in other_table:
                data_df = data_processing(data_df, ele_dict[element])
            else:
                data_df.set_index('Datetime', inplace=True)
                data_df.index = pd.DatetimeIndex(data_df.index)
                data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in data_df.columns:
                    data_df.drop(['Unnamed: 0'], axis=1, inplace=True)
            # 下载参考时段的数据
            if element in ele_caculate:
                sql_choose=sql_dict['year_cal']
            else:
                sql_choose=sql_dict['year']
                
            if element not in other_table:
                sql_choose=sql_dict['day']
                
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
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
            
            if element in other_table:
                refer_df = data_processing(refer_df, ele_dict[element])
            else:
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
                      
            if element in other_table:
                nearly_df = data_processing(nearly_df, ele_dict[element])
            else:
                nearly_df.set_index('Datetime', inplace=True)
                nearly_df.index = pd.DatetimeIndex(nearly_df.index)
                nearly_df['Station_Id_C'] = nearly_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in nearly_df.columns:
                    nearly_df.drop(['Unnamed: 0'], axis=1, inplace=True)
                    
        elif time_freq == 'D1': # '%Y%m%d,%Y%m%d'
        
            if element in ele_caculate:
                sql_choose=sql_dict['day_cal']
            else:
                sql_choose=sql_dict['day']
        
            if element in ele_day:
                ele_dict[element]=element
                
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            sta_ids = tuple(sta_ids.split(','))
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
                            WHERE
                                ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                                OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s))
                                AND station_id_c IN %s
                            """)
                            
                
            # 下载统计年份的数据
            start_year = stats_times.split(',')[0][:4]
            end_year = stats_times.split(',')[1][:4]
            start_month = stats_times.split(',')[0][4:6]
            end_month = stats_times.split(',')[1][4:6]
            start_date = stats_times.split(',')[0][6:]
            end_date = stats_times.split(',')[1][6:]
            
            cur.execute(query, (start_year, start_month, start_date, start_year, end_year, end_year, end_month, end_date, sta_ids))
            data = cur.fetchall()
            data_df = pd.DataFrame(data)
            data_df.columns = elements.split(',')
            if element in other_table:
                data_df = data_processing(data_df, ele_dict[element])
            else:
                data_df.set_index('Datetime', inplace=True)
                data_df.index = pd.DatetimeIndex(data_df.index)
                data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in data_df.columns:
                    data_df.drop(['Unnamed: 0'], axis=1, inplace=True)            
            
            if element in ele_caculate:
                sql_choose=sql_dict['year_cal']
            else:
                sql_choose=sql_dict['year']

            if element in ele_day:
                ele_dict[element]=element+'_Days'
                
            # 下载参考时段的数据
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' + ele_dict[element]
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
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
            
            if element in other_table:
                refer_df = data_processing(refer_df, ele_dict[element])
            else:
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
                      
            if element in other_table:
                nearly_df = data_processing(nearly_df, ele_dict[element])
            else:
                nearly_df.set_index('Datetime', inplace=True)
                nearly_df.index = pd.DatetimeIndex(nearly_df.index)
                nearly_df['Station_Id_C'] = nearly_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in nearly_df.columns:
                    nearly_df.drop(['Unnamed: 0'], axis=1, inplace=True)
                    
        elif time_freq == 'D2': # ['%Y,%Y','%m%d,%m%d']
        
            if element in ele_caculate:
                sql_choose=sql_dict['day_cal']
            else:
                sql_choose=sql_dict['day']
        
            if element in ele_day:
                ele_dict[element]=element
            
            sta_ids = tuple(sta_ids.split(','))
            elements = 'Station_Id_C,Station_Name,Datetime,' + ele_dict[element]
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
                            WHERE
                                (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND (
                                    (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                                    OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) < %s)
                                    OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s)
                                ))
                                AND station_id_c IN %s
                            """)

            # 下载统计年份的数据 ['%Y,%Y','%m%d,%m%d']
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
            data_df = pd.DataFrame(data)
            data_df.columns = elements.split(',')
            
            if element in other_table:
                data_df = data_processing(data_df, ele_dict[element])
            else:
                data_df.set_index('Datetime', inplace=True)
                data_df.index = pd.DatetimeIndex(data_df.index)
                data_df['Station_Id_C'] = data_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in data_df.columns:
                    data_df.drop(['Unnamed: 0'], axis=1, inplace=True)            
            
            if element in ele_caculate:
                sql_choose=sql_dict['year_cal']
            else:
                sql_choose=sql_dict['year']

            if element in ele_day:
                ele_dict[element]=element+'_Days'
                
            # 下载参考时段的数据
            elements = 'Station_Id_C,Station_Name,Lon,Lat,Alti,Datetime,' +ele_dict[element]
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{sql_choose}
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
            
            if element in other_table:
                refer_df = data_processing(refer_df, ele_dict[element])
            else:
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
                      
            if element in other_table:
                nearly_df = data_processing(nearly_df, ele_dict[element])
            else:
                nearly_df.set_index('Datetime', inplace=True)
                nearly_df.index = pd.DatetimeIndex(nearly_df.index)
                nearly_df['Station_Id_C'] = nearly_df['Station_Id_C'].astype(str)
    
                if 'Unnamed: 0' in nearly_df.columns:
                    nearly_df.drop(['Unnamed: 0'], axis=1, inplace=True)
                            
        # 关闭数据库
        cur.close()
        conn.close()


    # 开始计算
    # 首先获取站号对应的站名
    station_df = pd.DataFrame()
    station_df['站号'] = [
        51886, 51991, 52602, 52633, 52645, 52657, 52707, 52713, 52737, 52745, 52754, 52765, 52818, 52825, 52833, 52836, 52842, 52851, 52853, 52855, 52856, 52859, 52862, 52863, 52866, 52868, 52869, 52874, 52875, 52876, 52877, 52908, 52942, 52943,
        52955, 52957, 52963, 52968, 52972, 52974, 56004, 56015, 56016, 56018, 56021, 56029, 56033, 56034, 56043, 56045, 56046, 56065, 56067, 56125, 56151]
    station_df['站名'] = [
        '茫崖国家基准气候站', '那陵格勒国家基准气候站', '冷湖国家基准气候站', '托勒国家基本气象站', '野牛沟国家基准气候站', '祁连国家基本气象站', '小灶火国家基本气象站', '大柴旦国家基准气候站', '德令哈国家基本气象站', '天峻国家基本气象站', '刚察国家基准气候站', '门源国家基本气象站', '格尔木国家基准气候站', '诺木洪国家基准气候站', '乌兰国家基本气象站', '都兰国家基本气象站', '茶卡国家基准气候站', '江西沟国家基本气象站',
        '海晏国家基本气象站', '湟源国家基本气象站', '共和国家基本气象站', '瓦里关国家基本气象站', '大通国家基本气象站', '互助国家基本气象站', '西宁国家基本气象站', '贵德国家基本气象站', '湟中国家基本气象站', '乐都国家基本气象站', '平安国家基本气象站', '民和国家基准气候站', '化隆国家基本气象站', '五道梁国家基本气象站', '河卡国家基本气象站', '兴海国家基准气候站', '贵南国家基本气象站', '同德国家基本气象站',
        '尖扎国家基本气象站', '泽库国家基本气象站', '循化国家基本气象站', '同仁国家基本气象站', '沱沱河国家基准气候站', '曲麻河国家基准气候站', '治多国家基本气象站', '杂多国家基准气候站', '曲麻莱国家基本气象站', '玉树国家基本气象站', '玛多国家基准气候站', '清水河国家基本气象站', '玛沁国家基本气象站', '甘德国家基本气象站', '达日国家基准气候站', '河南国家基本气象站', '久治国家基准气候站', '囊谦国家基准气候站',
        '班玛国家基本气象站']
    station_df['站号'] = station_df['站号'].map(str)
    new_station = station_df[ station_df['站号'].isin(sta_ids)]

    # stats_result 展示结果表格
    # post_data_df 统计年份数据，用于后续计算
    # post_refer_df 参考年份数据，用于后续计算
    if element in tem_table:
        stats_result, post_data_df, post_refer_df, reg_params = tem_table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year,l_data=l_data,n_data=n_data)
    elif element in pre_table:
        stats_result, post_data_df, post_refer_df, reg_params = pre_table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year,R=R,R_flag=R_flag,RD=RD,RD_flag=RD_flag,Rxxday=Rxxday)

    elif element in other_table:
        stats_result, post_data_df, post_refer_df, reg_params = other_table_stats(data_df, refer_df, nearly_df, time_freq,element, last_year)
      
    print('1.统计表完成')
    
    # return stats_result, post_data_df, post_refer_df
    # 分布图
    nc_path, _, _, _, _ = contour_picture(stats_result, data_df, shp_path, interp_method, data_dir)
    nc_path_trans = nc_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
    nc_path_trans = nc_path_trans.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
    print('分布图插值生成nc完成')

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

    # 6. 统计分析-EOF分析
    ds = xr.open_dataset(nc_path)
    eof_path = eof(ds, shp_path, data_dir)
    print('eof完成')

    # 7. 统计分析-REOF分析
    ds = xr.open_dataset(nc_path)
    reof_path = reof(ds, shp_path, data_dir)
    print('reof完成')

    # 8.EEMD分析
    eemd_result = eemd(post_data_df, data_dir)
    print('eemd完成')

    # 数据保存

    result_dict = dict()
    result_dict['uuid'] = uuid4

    result_dict['表格'] = dict()
    result_dict['表格'] = stats_result.to_dict()

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
    data_json = dict()
    data_json['element'] ='drought'
    data_json['refer_years'] = '1991,2020'
    data_json['nearly_years'] = '2014,2021'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '1981,2023'
    data_json['sta_ids'] = '52754,56151,52855,52862,56065,52645,56046,52955,52968,52963,52825,56067,52713,52943,52877,52633,52866,52737,52745,52957,56018,56033,52657,52765,52972,52868,56016,52874,51886,56021,52876,56029,56125,52856,52836,52842,56004,52974,52863,56043,52908,56045,52818,56034,52853,52707,52602,52869,52833,52875,52859,52942,52851'
    data_json['interp_method'] = 'ukri'
    data_json['ci'] = 95
    data_json['shp_path'] =r'C:\Users\MJY\Desktop\qhbh\文档\03-边界矢量\03-边界矢量\03-边界矢量\08-省州界\省界.shp'
    
    result = extreme_climate_features(data_json)
    
    import simplejson

    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result['表格']}, ensure_ascii=False, ignore_nan=True)


