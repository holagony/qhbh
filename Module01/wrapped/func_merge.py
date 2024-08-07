import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Utils.utils import array2nc
from Module01.wrapped.table_stats import table_stats
from Module01.wrapped.contour_ficture import contour_picture

# step1 读取数据，计算基础的统计表格
path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
df = pd.read_csv(path, low_memory=False)
df = data_processing(df)
data_df = df[df.index.year <= 5000]
refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
nearly_df = df[df.index.year > 2011]
last_year = 2023
time_freq = 'M1'
ele = 'TEM_Avg'

# stats_result 展示结果表格
# post_data_df 统计年份数据，用于后续计算
# post_refer_df 参考年份数据，用于后续计算
stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)

# step2 分布图绘制
method = 'idw2'
shp_path = ''
output_filepath = r'c:\Users\MJY\Desktop\result'
result, data, gridx, gridy, year = contour_picture(stats_result, data_df, shp_path, method, output_filepath)

# step3 EOF/REOF
# numpy to xarary
var_name = '气温'
ds = array2nc(data, gridy, gridx, var_name, time=year)
