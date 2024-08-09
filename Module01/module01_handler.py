import os
import uuid
import pandas as pd
import xarray as xr
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.data_loader_with_threads import get_cmadaas_yearly_data
from Utils.data_loader_with_threads import get_cmadaas_monthly_data
from Utils.data_loader_with_threads import get_cmadaas_daily_data
from Utils.data_loader_with_threads import get_cmadaas_daily_period_data
from Utils.data_processing import data_processing
from Module01.wrapped.table_stats import table_stats
from Module01.wrapped.contour_ficture import contour_picture
from Module01.wrapped.mk_tests import time_analysis
from Module01.wrapped.cumsum_anomaly import calc_anomaly_cum
from Module01.wrapped.moving_avg import calc_moving_avg
from Module01.wrapped.wavelet_analyse import wavelet_main
from Module01.wrapped.correlation_analysis import correlation_analysis
from Module01.wrapped.eof import eof,reof
from Module01.wrapped.eemd import eemd


def statistical_climate_features(data_json):
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
    '''
    result_dict = edict()

    # 1.参数读取
    element = data_json['element']
    refer_years = data_json['refer_years']
    nearly_years = data_json['nearly_years']
    time_freq = data_json['time_freq']
    stats_times = data_json['stats_times']
    sta_ids = data_json['sta_ids']
    interp_method = data_json['interp_method']
    shp_path = data_json['shp_path']
    output_filepath = data_json['output_filepath']
    
    # 2.参数处理
    uuid4 = uuid.uuid4().hex
    result_dict['uuid'] = uuid4
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)

    if cfg.INFO.READ_LOCAL:

        # 3.解析要下载的参数
        last_year = int(nearly_years.split(',')[-1]) # 上一年的年份
        ele = ''
        elements_list = ['TEM_Avg', 'TEM_Max', 'TEM_Min', 'PRE_Time_2020', 'PRE_Days', 'PRE_Max_Day', 'PRS_Avg', 'PRS_Max', 'PRS_Min', 
                         'WIN_S_2mi_Avg', 'WIN_S_Max', 'WIN_S_Inst_Max', 'WIN_D_S_Max_C', 'GST_Avg', 'GST_Max', 'GST_Min', 'GST_Avg_5cm', 'GST_Avg_10cm', 
                         'GST_Avg_15cm', 'GST_Avg_20cm', 'GST_Avg_40cm', 'GST_Avg_80cm', 'GST_Avg_160cm', 'GST_Avg_320cm', 'CLO_Cov_Avg', 'CLO_Cov_Low_Avg', 
                         'SSH', 'SSP_Mon', 'EVP_Big', 'EVP', 'RHU_Avg', 'RHU_Min']
        resample_max = ['TEM_Max','PRS_Max','WIN_S_Max','WIN_S_Inst_Max','GST_Max']
        resample_min = ['TEM_Min','PRS_Min','GST_Min','RHU_Min']
        resample_sum = ['PRE_Time_2020','PRE_Days']
        resample_mean = ['TEM_Avg','PRS_Avg','WIN_S_2mi_Avg','WIN_D_S_Max_C','GST_Avg','GST_Avg_5cm','GST_Avg_10cm','GST_Avg_15cm','GST_Avg_20cm','GST_Avg_40cm', 
                         'GST_Avg_80cm','GST_Avg_160cm','GST_Avg_320cm','CLO_Cov_Avg','CLO_Cov_Low_Avg','SSH','SSP_Mon','EVP_Big','EVP','RHU_Avg']
    
        if element in elements_list:
            ele += element
    
        elif element == 'WIND':
            if time_freq in ['D1','D2']:
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
            data_df = data_processing(data_df)
    
            # 下载参考时段的数据
            refer_df = get_cmadaas_yearly_data(refer_years, ele, sta_ids)
            refer_df = data_processing(refer_df)
    
            # 下载近10年的数据
            nearly_df = get_cmadaas_yearly_data(nearly_years, ele, sta_ids)
            nearly_df = data_processing(nearly_df)
    
        elif time_freq == 'Q':
            # 统一使用的后处理pandas apply func
            ele_max = list(set(data_df.columns) & set(resample_max))
            ele_min = list(set(data_df.columns) & set(resample_min))
            ele_sum = list(set(data_df.columns) & set(resample_sum))
            ele_mean = list(set(data_df.columns) & set(resample_mean))
    
            def sample(x):
                x_info = x[['Station_Id_C','Station_Name','Lat','Lon','Year']].resample('1A').first()
                x_max = x[ele_max].resample('1A').max()
                x_min = x[ele_min].resample('1A').min()
                x_sum = x[ele_sum].resample('1A').sum()
                x_mean = x[ele_mean].resample('1A').mean().round(1)
                x_concat = pd.concat([x_info,x_max,x_min,x_sum,x_mean],axis=1)
                return x_concat
        
            # 下载统计年份的数据
            mon_list =  [int(mon_) for mon_ in stats_times[1].split(',')] # 提取月份
            years = stats_times[0]
            mon = '01,12'
            data_df = get_cmadaas_monthly_data(years, mon, ele, sta_ids)
            data_df = data_processing(data_df)
    
            # TODO if element in ['EVP_Penman', 'EVP_taka']:
            data_df = data_df[data_df['Mon'].isin(mon_list)]
            data_df = data_df.groupby('Station_Id_C').apply(sample) # 转化为季度数据
            data_df.reset_index(level=0,drop=True,inplace=True)
    
            # 下载参考时段的数据
            refer_df = get_cmadaas_monthly_data(refer_years, mon, ele, sta_ids)
            refer_df = data_processing(refer_df)
            refer_df = refer_df[refer_df['Mon'].isin(mon_list)]
            refer_df = refer_df.groupby('Station_Id_C').apply(sample) # 转化为季度数据
            refer_df.reset_index(level=0,drop=True,inplace=True)
    
            # 下载近10年的数据
            nearly_df = get_cmadaas_monthly_data(nearly_years, mon, ele, sta_ids)
            nearly_df = data_processing(nearly_df)
            nearly_df = nearly_df[nearly_df['Mon'].isin(mon_list)]
            nearly_df = nearly_df.groupby('Station_Id_C').apply(sample) # 转化为季度数据
            nearly_df.reset_index(level=0,drop=True,inplace=True)
    
        elif time_freq == 'M1':
            # 下载统计年份的数据
            start_time = stats_times.split(',')[0]
            end_time = stats_times.split(',')[1]
            years = start_time[:4] + ',' + end_time[:4]
            mon = start_time[4:] + ',' + end_time[4:]
            data_df = get_cmadaas_monthly_data(years, mon, ele, sta_ids)
            data_df = data_processing(data_df)
    
            # 下载参考时段的数据
            refer_df = get_cmadaas_monthly_data(refer_years, mon, ele, sta_ids)
            refer_df = data_processing(refer_df)
    
            # 下载近10年的数据
            nearly_df = get_cmadaas_monthly_data(nearly_years, mon, ele, sta_ids)
            nearly_df = data_processing(nearly_df)
    
        elif time_freq == 'M2':
            # 下载统计年份的数据
            mon_list =  [int(mon_) for mon_ in stats_times[1].split(',')]
            years = stats_times[0]
            mon = '01,12'
            data_df = get_cmadaas_monthly_data(years, mon, ele, sta_ids)
            data_df = data_processing(data_df)
            data_df = data_df[data_df['Mon'].isin(mon_list)] # 按区间提取月份
    
            # 下载参考时段的数据
            refer_df = get_cmadaas_monthly_data(refer_years, mon, ele, sta_ids)
            refer_df = data_processing(refer_df)
            refer_df = refer_df[refer_df['Mon'].isin(mon_list)]
    
            # 下载近10年的数据
            nearly_df = get_cmadaas_monthly_data(nearly_years, mon, ele, sta_ids)
            nearly_df = data_processing(nearly_df)
            nearly_df = nearly_df[nearly_df['Mon'].isin(mon_list)]
    
        elif time_freq == 'D1':
            # 下载统计年份的数据
            start_time = time_freq.split(',')[0]
            end_time = time_freq.split(',')[1]
            years = start_time[:4] + ',' + end_time[:4]
            date = start_time[4:] + ',' + end_time[4:]
            data_df = get_cmadaas_daily_data(years, date, ele, sta_ids)
            data_df = data_processing(data_df)
    
            # 下载参考时段的数据
            refer_df = get_cmadaas_daily_data(refer_years, date, ele, sta_ids)
            refer_df = data_processing(refer_df)
    
            # 下载近10年的数据
            nearly_df = get_cmadaas_daily_data(nearly_years, date, ele, sta_ids)
            nearly_df = data_processing(nearly_df)
    
        elif time_freq == 'D2':
            # 下载统计年份的数据
            years = time_freq[0]
            date = time_freq[1]
            data_df = get_cmadaas_daily_period_data(years, date, ele, sta_ids)
            data_df = data_processing(data_df)
    
            # 下载参考时段的数据
            refer_df = get_cmadaas_daily_period_data(refer_years, date, ele, sta_ids)
            refer_df = data_processing(refer_df)
    
            # 下载近10年的数据
            nearly_df = get_cmadaas_daily_period_data(nearly_years, date, ele, sta_ids)
            nearly_df = data_processing(nearly_df)
    else:
        path = r'D:\Project\3_项目\2_气候评估和气候可行性论证\qhkxxlz\Files\test_data\qh_mon.csv'

        df = pd.read_csv(path, low_memory=False)
        df = data_processing(df)
        data_df = df[df.index.year <= 5000]
        refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
        nearly_df = df[df.index.year > 2011]
        last_year = 2023



    # stats_result 展示结果表格
    # post_data_df 统计年份数据，用于后续计算
    # post_refer_df 参考年份数据，用于后续计算
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)
    
    # 分布图
    result, data, gridx, gridy, year = contour_picture(stats_result, data_df, shp_path, interp_method, output_filepath)
    
    # 1.统计分析-mk检验
    mk_result = time_analysis(post_data_df)
    
    # 2.统计分析-累积距平
    anomaly, anomaly_accum = calc_anomaly_cum(post_data_df, post_refer_df)
    
    # 3.统计分析-滑动平均
    moving_result = calc_moving_avg(post_data_df, 3)
    
    # 4. 统计分析-小波分析
    wave_result=wavelet_main(stats_result,output_filepath)
    
    # 5. 统计分析-相关分析
    correlation_result= correlation_analysis(post_data_df,output_filepath)
    
    # 6. 统计分析-EOF分析
    ds = xr.open_dataset(result)
    eof_path=eof(ds,shp_path,output_filepath)
    
    # 7. 统计分析-REOF分析
    ds = xr.open_dataset(result)
    reof_path=reof(ds,shp_path,output_filepath)
    
    # 8.EEMD分析
    eemd_result=eemd(stats_result,output_filepath)
    
    # 数据保存
    
    result_dict['表格']=dict()
    result_dict['表格']=stats_result.to_dict()
    
    result_dict['分布图']=dict()
    result_dict['分布图']=result
    
    result_dict['统计分析']=dict()
    result_dict['统计分析']['mk检验']=mk_result
    result_dict['统计分析']['累积距平']=dict()
    result_dict['统计分析']['累积距平']['距平']=anomaly.to_dict()
    result_dict['统计分析']['累积距平']['累积']=anomaly_accum.to_dict()
    result_dict['统计分析']['滑动平均']=moving_result.to_dict()
    result_dict['统计分析']['小波分析']=wave_result
    result_dict['统计分析']['相关分析']=correlation_result
    result_dict['统计分析']['EOF分析']=eof_path
    result_dict['统计分析']['REOF分析']=reof_path
    result_dict['统计分析']['EEMD分析']=eemd_result
    
    # end_time=time.perf_counter()
    # print(str(round(end_time-start_time,3))+'s')

    
    return result_dict
