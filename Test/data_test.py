import numpy as np
import pandas as pd
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

data_dir = r'C:/Users/MJY/Desktop/result'

# AGME_CHN_CROP_GROWTH
path = r'C:/Users/MJY/Desktop/data/TXT_20240801114250.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df = df[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Datetime','Year', 'Mon', 'Day', 'Crop_Name', 'GroPer_Name_Ten']]
df.sort_values('Datetime',inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])

df.set_index('Datetime', inplace=True, drop=False)
df = df[df['Crop_Name'].isin([10101,10201,10202,10203,10301,10401,10501,10601,10701,19999])]

# 返青期
df1 = df[df['GroPer_Name_Ten'].isin([21])]
df1 = df1[~df1.index.duplicated()]
df1['fanqing'] = df1.index.dayofyear


# In[]
# 黄枯期
df2 = df[df['GroPer_Name_Ten'].isin([91])]
df2 = df2[~df2.index.duplicated()]
df2['huangku'] = df2.index.dayofyear

df3 = data_processing(df2, 'huangku')

stats_result, post_data_df, post_refer_df, reg_params = table_stats(df3, df3, df3, 'huangku', 2020)


# AGME_CHN_GRASS_COVER 草地覆盖度
path = r'C:/Users/MJY/Desktop/data/TXT_20240801162359.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df = df[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Datetime','Year', 'Mon', 'Day', 'Cov']]
df1 = data_processing(df, 'Cov')
stats_result, post_data_df, post_refer_df, reg_params = table_stats(df1, df1, df1, 'Cov', 2020)

# In[]
# AGME_CHN_GRASS_HEIGH 草高 要素对不上
path = r'C:/Users/MJY/Desktop/data/TXT_20240801100828.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df = df[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Datetime','Year', 'Mon', 'Day', 'Crop_LiStoc_Name', 'Crop_Heigh']]
df.sort_values('Datetime',inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])



# In[]

# AGME_GRASS_HERBAGE_GROWTH_HEIGHT
path = r'C:/Users/MJY/Desktop/data/TXT_20240801144127.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['Station_Id_C']==56065]


# AGME_CHN_GRASS_YIELD
path = r'C:/Users/MJY/Desktop/data/TXT_20240801113526.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['Station_Id_C']==56065]

# In[]
# OTHE_METE_RIVER_QH
path = r'C:/Users/MJY/Desktop/data/TXT_20240801165537.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['STCD']==40100350]

# In[]
# OTHE_HYDR_RSVR_QH
path = r'C:/Users/MJY/Desktop/data/TXT_20240801165717.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['STCD']==40202210]


