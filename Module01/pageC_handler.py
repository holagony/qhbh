import os
import uuid
import time
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

# 草地植被

def grass_features_stats(data_json):
    '''
    获取天擎数据，参数说明
    :param element：对应原型，传入的要素名称
        草地返青期 grassland_green_period
        草地枯黄期 grassland_yellow_period
        草地生育期 grassland_growth_period
        草地覆盖度 grassland_coverage
        草高 grass_height
        草地产量 grassland_yield
        植被生态指数 vegetation_index
        植被净初级生产力 vegetation_pri_productivity
        植被覆盖度 vegetation_coverage
        植被固定CO2 vegetation_carbon
        湿地植被碳储量 wet_vegetation_carbon
        荒漠化等级 desert_level
        荒漠化面积 desert_area

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

    # 2.参数处理
    degree = None
    if shp_path is not None:
        shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径
        
    last_year = int(nearly_years.split(',')[-1])  # 上一年的年份

    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)

    # 确定表名
    table_dict = dict()
    table_dict['grassland_green_period'] = 'qh_climate_crop_growth'
    table_dict['grassland_yellow_period'] = 'qh_climate_crop_growth'
    table_dict['grassland_growth_period'] = '待定'
    table_dict['grassland_coverage'] = 'qh_climate_grass_cover'
    table_dict['grass_height'] = 'qh_climate_grass_height'
    table_dict['dwei'] = 'qh_climate_grass_yield'  # 草地产量干重
    table_dict['fwei'] = 'qh_climate_grass_yield'  # 草地产量湿重
    table_dict['vegetation_index'] = '待定'
    table_dict['vegetation_pri_productivity'] = '待定'
    table_dict['vegetation_coverage'] = '待定'
    table_dict['vegetation_carbon'] = '待定'
    table_dict['wet_vegetation_carbon'] = '待定'
    table_dict['desert_level'] = '待定'
    table_dict['desert_area'] = '待定'
    table_name = table_dict[element]

    # 确定要素
    element_dict = dict()
    element_dict['grassland_green_period'] = 'Crop_Name,GroPer_Name_Ten'
    element_dict['grassland_yellow_period'] = 'Crop_Name,GroPer_Name_Ten'
    element_dict['grassland_growth_period'] = '待定'
    element_dict['grassland_coverage'] = 'Cov'
    element_dict['grass_height'] = 'Crop_Heigh'
    element_dict['dwei'] = 'dwei'
    element_dict['fwei'] = 'fwei'
    element_dict['vegetation_index'] = '待定'
    element_dict['vegetation_pri_productivity'] = '待定'
    element_dict['vegetation_coverage'] = '待定'
    element_dict['vegetation_carbon'] = '待定'
    element_dict['wet_vegetation_carbon'] = '待定'
    element_dict['desert_level'] = '待定'
    element_dict['desert_area'] = '待定'
    element_str = element_dict[element]

    if not cfg.INFO.READ_LOCAL:
        pass
    else:  # 走数据库
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

        # 下载近10年的数据
        start_year = nearly_years.split(',')[0]
        end_year = nearly_years.split(',')[1]
        cur.execute(query, (start_year, end_year, sta_ids))
        data = cur.fetchall()
        nearly_df = pd.DataFrame(data)
        nearly_df.columns = elements.split(',')

        # 数据处理
        # 二次计算处理
        if element == 'grassland_green_period':
            # 一年一个记录，应该不用resample('1A')
            data_df['Crop_Name'] = data_df['Crop_Name'].map(int)
            data_df['Datetime'] = pd.to_datetime(data_df['Datetime'])
            data_df.set_index('Datetime', inplace=True, drop=False)
            data_df = data_df[data_df['Crop_Name'].isin([10101, 10201, 10202, 10203, 10301, 10401, 10501, 10601, 10701, 19999])]
            data_df = data_df[data_df['GroPer_Name_Ten'].isin(['21'])]  # 21是返青
            data_df = data_df[~data_df.index.duplicated()]
            data_df['fanqing'] = data_df.index.dayofyear
            data_df['fanqing_date'] = data_df.index.year

            refer_df['Crop_Name'] = refer_df['Crop_Name'].map(int)
            refer_df['Datetime'] = pd.to_datetime(refer_df['Datetime'])
            refer_df.set_index('Datetime', inplace=True, drop=False)
            refer_df = refer_df[refer_df['Crop_Name'].isin([10101, 10201, 10202, 10203, 10301, 10401, 10501, 10601, 10701, 19999])]
            refer_df = refer_df[refer_df['GroPer_Name_Ten'].isin(['21'])]  # 21是返青
            refer_df = refer_df[~refer_df.index.duplicated()]
            refer_df['fanqing'] = refer_df.index.dayofyear
            refer_df['fanqing_date'] = refer_df.index.year

            nearly_df['Crop_Name'] = nearly_df['Crop_Name'].map(int)
            nearly_df['Datetime'] = pd.to_datetime(nearly_df['Datetime'])
            nearly_df.set_index('Datetime', inplace=True, drop=False)
            nearly_df = nearly_df[nearly_df['Crop_Name'].isin([10101, 10201, 10202, 10203, 10301, 10401, 10501, 10601, 10701, 19999])]
            nearly_df = nearly_df[nearly_df['GroPer_Name_Ten'].isin(['21'])]  # 21是返青
            nearly_df = nearly_df[~nearly_df.index.duplicated()]
            nearly_df['fanqing'] = nearly_df.index.dayofyear
            nearly_df['fanqing_date'] = nearly_df.index.year
            element_str = 'fanqing'

        elif element == 'grassland_yellow_period':
            # 一年一个记录，应该不用resample('1A')
            data_df['Crop_Name'] = data_df['Crop_Name'].map(int)
            data_df['Datetime'] = pd.to_datetime(data_df['Datetime'])
            data_df.set_index('Datetime', inplace=True, drop=False)
            data_df = data_df[data_df['Crop_Name'].isin([10101, 10201, 10202, 10203, 10301, 10401, 10501, 10601, 10701, 19999])]
            data_df = data_df[data_df['GroPer_Name_Ten'].isin(['91'])]  # 21是返青
            data_df = data_df[~data_df.index.duplicated()]
            data_df['huangku'] = data_df.index.dayofyear
            data_df['huangku_date'] = data_df.index.year

            refer_df['Crop_Name'] = refer_df['Crop_Name'].map(int)
            refer_df['Datetime'] = pd.to_datetime(refer_df['Datetime'])
            refer_df.set_index('Datetime', inplace=True, drop=False)
            refer_df = refer_df[refer_df['Crop_Name'].isin([10101, 10201, 10202, 10203, 10301, 10401, 10501, 10601, 10701, 19999])]
            refer_df = refer_df[refer_df['GroPer_Name_Ten'].isin(['91'])]  # 21是返青
            refer_df = refer_df[~refer_df.index.duplicated()]
            refer_df['huangku'] = refer_df.index.dayofyear
            refer_df['huangku_date'] = refer_df.index.year

            nearly_df['Crop_Name'] = nearly_df['Crop_Name'].map(int)
            nearly_df['Datetime'] = pd.to_datetime(nearly_df['Datetime'])
            nearly_df.set_index('Datetime', inplace=True, drop=False)
            nearly_df = nearly_df[nearly_df['Crop_Name'].isin([10101, 10201, 10202, 10203, 10301, 10401, 10501, 10601, 10701, 19999])]
            nearly_df = nearly_df[nearly_df['GroPer_Name_Ten'].isin(['91'])]  # 21是返青
            nearly_df = nearly_df[~nearly_df.index.duplicated()]
            nearly_df['huangku'] = nearly_df.index.dayofyear
            nearly_df['huangku_date'] = nearly_df.index.year
            element_str = 'huangku'

        else:
            data_df = data_processing(data_df, element_str, degree)
            refer_df = data_processing(refer_df, element_str, degree)
            nearly_df = data_processing(nearly_df, element_str, degree)

        # 关闭数据库
        cur.close()
        conn.close()

    ######################################
    # 开始计算
    # 首先获取站号对应的站名
    station_df = pd.DataFrame()
    station_df['站号'] = [52955, 56080, 56079, 56074, 56065, 56045, 56021, 54102, 53821, 53817, 53723, 53644, 53505, 53384, 53289, 53231, 52943, 52876, 52869, 
                        52868, 52863, 52862, 52856, 52855, 52852, 52825, 52818, 52765, 52737, 52681, 52101, 51711, 51469, 51437, 50954, 50936, 50928, 50854, 
                        50742, 50618, 50525, 50425]
    station_df['站名'] = ['贵南', '合作', '若尔盖', '玛曲', '河南', '甘德', '曲麻莱', '锡林浩特', '环县', '固原', '盐池', '乌审旗', '孪井滩', '察哈尔右翼后旗', '镶黄旗', 
                        '海力素', '兴海', '民和', '湟中', '贵德', '互助', '大通', '共和', '湟源', '海北', '诺木洪', '格尔木', '门源', '德令哈', '民勤', '巴里坤', '阿合奇', 
                        '牧业', '昭苏', '肇源', '白城', '巴雅尔吐胡硕', '安达', '富裕', '新巴尔虎左旗', '鄂温克', '额尔古纳']
    station_df['站号'] = station_df['站号'].map(str)
    new_station = station_df[station_df['站号'].isin(sta_ids)]

    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['表格'] = dict()
    result_dict['分布图'] = dict()
    result_dict['统计分析'] = dict()
    result_dict['站号'] = new_station.to_dict(orient='records')


    # stats_result 展示结果表格
    # post_data_df 统计年份数据，用于后续计算
    # post_refer_df 参考年份数据，用于后续计算
    stats_result, post_data_df, post_refer_df, reg_params = table_stats(data_df, refer_df, nearly_df, element_str, last_year)
    result_dict['表格'] = stats_result.to_dict(orient='records')
    result_dict['统计分析']['线性回归'] = reg_params.to_dict(orient='records')
    print('统计表完成')

    # 分布图 try在里面了
    if shp_path is not None:
        nc_path, _, _, _, _ = contour_picture(stats_result, data_df, shp_path, interp_method, data_dir)
        nc_path_trans = nc_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
        nc_path_trans = nc_path_trans.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        print('分布图插值生成nc完成')
    else:
        nc_path = None
        nc_path_trans = None
    result_dict['分布图'] = nc_path_trans

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
        result_dict['统计分析']['EOF分析'] = eof_path
        result_dict['统计分析']['REOF分析'] = reof_path

    # 测试下来，只有1个值也能出结果，以下所有的暂时不用加异常处理
    # 1.统计分析-mk检验
    mk_result = time_analysis(post_data_df, data_dir) # 里面有try
    result_dict['统计分析']['MK检验'] = mk_result
    print('MK检验完成')

    # 2.统计分析-累积距平
    anomaly_result = calc_anomaly_cum(post_data_df, post_refer_df, data_dir)
    result_dict['统计分析']['累积距平'] = anomaly_result
    print('距平完成')

    # 3.统计分析-滑动平均
    moving_result = calc_moving_avg(post_data_df, 5, data_dir)
    result_dict['统计分析']['滑动平均'] = moving_result
    print('滑动平均完成')

    # 4. 统计分析-小波分析
    wave_result = wavelet_main(post_data_df, data_dir)
    result_dict['统计分析']['小波分析'] = wave_result
    print('小波完成')

    # 5. 统计分析-相关分析
    correlation_result = correlation_analysis(post_data_df, data_dir)
    result_dict['统计分析']['相关分析'] = correlation_result
    print('相关分析完成')
    
    # 8.EEMD分析
    eemd_result = eemd(post_data_df, data_dir)
    result_dict['统计分析']['EEMD分析'] = eemd_result
    print('eemd完成')

    # 8.EEMD分析
    eemd_result = eemd(post_data_df, data_dir)
    print('eemd完成')

    return result_dict


if __name__ == '__main__':
    t1 = time.time()
    data_json = dict()
    data_json['element'] = 'dwei'
    data_json['refer_years'] = '1991,2020'
    data_json['nearly_years'] = '2014,2023'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '1981,2023'
    data_json['sta_ids'] = '52943,56021,56045,56065'
    data_json['interp_method'] = 'ukri'
    data_json['ci'] = 95
    data_json['shp_path'] = r'C:\Users\MJY\Desktop\qhbh\文档\03-边界矢量\03-边界矢量\03-边界矢量\01-青海省\青海省县级数据.shp'

    result = grass_features_stats(data_json)
    t2 = time.time()
    print(t2 - t1)
