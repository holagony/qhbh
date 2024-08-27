import numpy as np
import pandas as pd
from Utils.data_processing import data_processing

# AGME_CHN_CROP_GROWTH
path = r'C:/Users/MJY/Desktop/data/TXT_20240801114250.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['Station_Id_C']==52856]


# AGME_CHN_GRASS_COVER
path = r'C:/Users/MJY/Desktop/data/TXT_20240801162359.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df = df[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Datetime','Year', 'Mon', 'Day', 'Cov']]
# df = df[df['Station_Id_C']==56065]
df1 = data_processing(df, 'Cov')




# In[]

# AGME_CHN_GRASS_HEIGH
path = r'C:/Users/MJY/Desktop/data/TXT_20240801100828.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['Station_Id_C']==56065]


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


