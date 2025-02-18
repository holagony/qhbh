import os
import uuid
import numpy as np
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
from Utils.data_loader_with_threads import get_database_data

# 气候要素

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
        辐射 Rad

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

    try:
        last_year = int(nearly_years.split(',')[-1])  # 上一年的年份
    except:
        last_year = int(nearly_years[0].split(',')[-1])
        
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)
    
    # 确定表名
    if time_freq == 'Y':
        table_name = 'qh_climate_cmadaas_year'
    elif time_freq in ['Q', 'M1', 'M2']:
        table_name = 'qh_climate_cmadaas_month'
    elif time_freq in ['D1', 'D2']:
        table_name = 'qh_climate_cmadaas_day'
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
    
    if element == 'EVP_Taka':
        table_name = 'qh_qhbh_cmadaas_month'
        element_str = 'TEM_Avg,PRE_Time_2020'
    
    if element == 'Rad':
        table_name = 'qh_climate_radi_hour'
        element_str = 'V14311'

    sta_ids = tuple(sta_ids.split(','))
    
    # 从数据库截数据
    data_df = get_database_data(sta_ids, element_str, table_name, time_freq, stats_times)
    refer_df = get_database_data(sta_ids, element_str, table_name, time_freq, refer_years)
    nearly_df = get_database_data(sta_ids, element_str, table_name, time_freq, nearly_years)

    # 数据处理
    if element == 'EVP_Taka':
        data_df['EVP_Taka'] = 3100*data_df['TEM_Avg']/(3100+1.8*(data_df['PRE_Time_2020']**2)*np.exp((-34.4*data_df['TEM_Avg'])/(235+data_df['TEM_Avg'])))
        refer_df['EVP_Taka'] = 3100*refer_df['TEM_Avg']/(3100+1.8*(refer_df['PRE_Time_2020']**2)*np.exp((-34.4*refer_df['TEM_Avg'])/(235+refer_df['TEM_Avg'])))
        nearly_df['EVP_Taka'] = 3100*nearly_df['TEM_Avg']/(3100+1.8*(nearly_df['PRE_Time_2020']**2)*np.exp((-34.4*nearly_df['TEM_Avg'])/(235+nearly_df['TEM_Avg'])))

        data_df['EVP_Taka'] = np.where(data_df['EVP_Taka']<0,0,data_df['EVP_Taka'])
        refer_df['EVP_Taka'] = np.where(refer_df['EVP_Taka']<0,0,refer_df['EVP_Taka'])
        nearly_df['EVP_Taka'] = np.where(nearly_df['EVP_Taka']<0,0,nearly_df['EVP_Taka'])

        element_str = 'EVP_Taka'

    data_df = data_processing(data_df, element_str, degree)
    refer_df = data_processing(refer_df, element_str, degree)
    nearly_df = data_processing(nearly_df, element_str, degree)
    
    if element == 'Rad':
        data_df['V14311'] = data_df['V14311'].map(float)*3600/1e6
        refer_df['V14311'] = refer_df['V14311'].map(float)*3600/1e6
        nearly_df['V14311'] = nearly_df['V14311'].map(float)*3600/1e6

    ######################################################
    # 开始计算
    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['表格'] = dict()
    result_dict['分布图'] = dict()
    result_dict['统计分析'] = dict()

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
    result_dict['站号'] = new_station.to_dict(orient='records')

    # 如果是积温，此时的element_str是TEM_Avg，需要修改为Accum_Tem
    if element == 'Accum_Tem':
        element_str = 'Accum_Tem'

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

    return result_dict


if __name__ == '__main__':
    t1 = time.time()
    data_json = dict()
    data_json['element'] = 'TEM_Avg'
    data_json['refer_years'] = '1994,2023'
    data_json['nearly_years'] = '1994,2022'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2011,2021'  # '198105,202009' # '1981,2023'
    data_json['sta_ids'] = '52866,56029,52863,52754,52818,52874,56043,52713,56065'
    data_json['interp_method'] = 'idw2'
    data_json['ci'] = 95
    data_json['shp_path'] =r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\省界.shp'
    data_json['degree'] = None
    
    result = climate_features_stats(data_json)
    t2 = time.time()
    print(t2 - t1)
