import numpy as np
import pandas as pd
import multiprocessing
import time
import xarray as xr
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Utils.utils import array2nc
from Module01.wrapped.table_stats import table_stats
from Module01.wrapped.contour_ficture import contour_picture
from Module01.wrapped.mk_tests import time_analysis
from Module01.wrapped.cumsum_anomaly import calc_anomaly_cum
from Module01.wrapped.moving_avg import calc_moving_avg
from Module01.wrapped.moving_avg import calc_moving_avg
from Module01.wrapped.wavelet_analyse import wavelet_main
from Module01.wrapped.correlation_analysis import correlation_analysis
from Module01.wrapped.eof import eof,reof
from Module01.wrapped.eemd import eemd


# step1 读取数据，计算基础的统计表格
path = r'D:\Project\3_项目\2_气候评估和气候可行性论证\qhkxxlz\Files\test_data\qh_mon.csv'
shp_path = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\省界.shp'
output_filepath = r'D:\Project\qh\2'

df = pd.read_csv(path, low_memory=False)
df = data_processing(df)
data_df = df[df.index.year <= 5000]
refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
nearly_df = df[df.index.year > 2011]
last_year = 2023
time_freq = 'M1'
ele = 'TEM_Avg'
method = 'idw2'




start_time=time.perf_counter()


# stats_result 展示结果表格
# post_data_df 统计年份数据，用于后续计算
# post_refer_df 参考年份数据，用于后续计算
stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)

# 分布图
result, data, gridx, gridy, year = contour_picture(stats_result, data_df, shp_path, method, output_filepath)

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
result_df=dict()

result_df['表格']=dict()
result_df['表格']=stats_result.to_dict()

result_df['分布图']=dict()
result_df['分布图']=result

result_df['统计分析']=dict()
result_df['统计分析']['mk检验']=mk_result
result_df['统计分析']['累积距平']=dict()
result_df['统计分析']['累积距平']['距平']=anomaly.to_dict()
result_df['统计分析']['累积距平']['累积']=anomaly_accum.to_dict()
result_df['统计分析']['滑动平均']=moving_result.to_dict()
result_df['统计分析']['小波分析']=wave_result
result_df['统计分析']['相关分析']=correlation_result
result_df['统计分析']['EOF分析']=eof_path
result_df['统计分析']['REOF分析']=reof_path
result_df['统计分析']['EEMD分析']=eemd_result





end_time=time.perf_counter()
print(str(round(end_time-start_time,3))+'s')
