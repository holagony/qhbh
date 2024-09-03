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
from Module01.wrapped.func01_table_stats import table_stats
from Module01.wrapped.func02_interp_grid import contour_picture
from Module01.wrapped.func03_mk_tests import time_analysis
from Module01.wrapped.func04_cumsum_anomaly import calc_anomaly_cum
from Module01.wrapped.func05_moving_avg import calc_moving_avg
from Module01.wrapped.func06_wavelet_analyse import wavelet_main
from Module01.wrapped.func07_correlation_analysis import correlation_analysis
from Module01.wrapped.func08_eof import eof, reof
from Module01.wrapped.func09_eemd import eemd
import time


def climate_features_stats(data_json):
    '''
    获取天擎数据，参数说明
    :param element：对应原型，传入的要素名称
        平均气温	TEM_Avg 
        最高气温	TEM_Max
        最低气温	TEM_Min
        降水量	PRE_Time_2020
        降水日数	PRE_Days (日没有)
        年最大日降水	PRE_Max_Day (只有年有)
        平均气压	PRS_Avg
        最高气压	PRS_Max
        最低气压	PRS_Min
        平均风速	WIN_S_2mi_Avg
        最大风速	WIN_S_Max
        极大风速	WIN_S_Inst_Max
        日最大风速风向	WIN_D_S_Max (日) / WIN_D_S_Max_C (月和年)
        平均地面温度	GST_Avg
        最高地面温度	GST_Max
        最低地面温度	GST_Min
        平均5cm地温 GST_Avg_5cm
        平均10cm地温	GST_Avg_10cm
        平均15cm地温	GST_Avg_15cm
        平均20cm地温	GST_Avg_20cm
        平均40cm地温	GST_Avg_40cm
        平均80cm地温	GST_Avg_80cm
        平均160cm地温	GST_Avg_160cm
        平均320cm地温	GST_Avg_320cm
        平均总云量	CLO_Cov_Avg
        平均低云量	CLO_Cov_Low_Avg
        日照时数	SSH
        月日照百分率	SSP_Mon (日没有)
        大蒸发	EVP_Big
        小蒸发	EVP
        高桥蒸发	EVP_Taka
        彭曼蒸发	EVP_Penman
        平均相对湿度	RHU_Avg
        最小相对湿度	RHU_Min
        积温    Accum_Tem

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
    
    # 确定表名
    if time_freq == 'Y':
        table_name = 'qh_qhbh_cmadaas_year'
    elif time_freq in ['Q', 'M1', 'M2']:
        table_name = 'qh_qhbh_cmadaas_month'
    elif time_freq in ['D1', 'D2']:
        table_name = 'qh_qhbh_cmadaas_day'
    element_str = element
    
    if element == 'EVP_Penman' and time_freq == 'Y':
        table_name = 'qh_qhbh_calc_elements_year'
        element_str = 'pmet'
    elif element == 'EVP_Penman' and time_freq in ['Q', 'M1', 'M2']:
        table_name = 'qh_qhbh_calc_elements_month'
        element_str = 'pmet'
    elif element == 'EVP_Penman' and time_freq in ['D1', 'D2']:
        table_name = 'qh_qhbh_calc_elements_day'
        element_str = 'pmet'
    
    if element == 'Accum_Tem': # 积温
        table_name = 'qh_qhbh_cmadaas_day'
        element_str = 'TEM_Avg'

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
                            ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s))
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
    data_df = data_processing(data_df, element_str, degree)
    
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
    refer_df = data_processing(refer_df, element_str, degree)

    # 下载近10年的数据
    start_year = nearly_years.split(',')[0]
    end_year = nearly_years.split(',')[1]
    cur.execute(query, (start_year, end_year, sta_ids))
    data = cur.fetchall()
    nearly_df = pd.DataFrame(data)
    nearly_df.columns = elements.split(',')
    nearly_df = data_processing(nearly_df, element_str, degree)

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

    # 如果是积温，此时的element_str是TEM_Avg，需要修改为Accum_Tem
    if element == 'Accum_Tem':
        element_str = 'Accum_Tem'

    stats_result, post_data_df, post_refer_df, reg_params = table_stats(data_df, refer_df, nearly_df, element_str, last_year)
    print('统计表完成')

    # 分布图
    if shp_path is not None:
        nc_path, _, _, _, _ = contour_picture(stats_result, data_df, shp_path, interp_method, data_dir)
        nc_path_trans = nc_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
        nc_path_trans = nc_path_trans.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        print('分布图插值生成nc完成')
    else:
        nc_path = None
        nc_path_trans = None
        print('没有shp文件，散点图，生成nc')

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

    # 6/7. 统计分析-EOF分析
    if nc_path is not None:
        ds = xr.open_dataset(nc_path)
        eof_path = eof(ds, shp_path, data_dir)
        print('eof完成')
        reof_path = reof(ds, shp_path, data_dir)
        print('reof完成')
    else:
        eof_path = None
        reof_path = None
        print('没有插值生成网格文件，无法计算eof/reof')

    # 8.EEMD分析
    eemd_result = eemd(post_data_df, data_dir)
    print('eemd完成')

    # 数据保存
    result_dict = dict()
    result_dict['uuid'] = uuid4

    result_dict['表格'] = dict()
    result_dict['表格'] = stats_result.to_dict(orient='records')

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
    data_json['element'] = 'Accum_Tem'
    data_json['refer_years'] = '1991,2020'
    data_json['nearly_years'] = '2014,2023'
    data_json['time_freq'] = 'Q'
    data_json['stats_times'] = ['1981,2020', '3,4,5']  # '198105,202009' # '1981,2023'
    data_json['sta_ids'] = '52754,56151,52855,52862,56065,52645,56046,52955,52968,52963,52825,56067,52713,52943,52877,52633,52866,52737,52745,52957,56018,56033,52657,52765,52972,52868,56016,52874,51886,56021,52876,56029,56125,52856,52836,52842,56004,52974,52863,56043,52908,56045,52818,56034,52853,52707,52602,52869,52833,52875,52859,52942,52851'
    data_json['interp_method'] = 'ukri'
    data_json['ci'] = 95
    data_json['shp_path'] = r'C:\Users\MJY\Desktop\qhbh\文档\03-边界矢量\03-边界矢量\03-边界矢量\01-青海省\青海省县级数据.shp'
    data_json['degree'] = 10
    
    result = climate_features_stats(data_json)
    t2 = time.time()
    print(t2 - t1)
