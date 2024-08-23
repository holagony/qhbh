import numpy as np
import pandas as pd


# AGME_CHN_CROP_GROWTH
path = r'C:/Users/MJY/Desktop/data/TXT_20240801114250.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['Station_Id_C']==52856]
df = df[df['GroPer_Name_Ten']==91]


# AGME_CHN_GRASS_COVER
path = r'C:/Users/MJY/Desktop/data/TXT_20240801162359.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['Station_Id_C']==56065]


# AGME_CHN_GRASS_HEIGH
path = r'C:/Users/MJY/Desktop/data/TXT_20240801100828.txt'
df = pd.read_csv(path,encoding='gbk',sep='\t')
df.sort_values('Datetime',inplace=True)
df = df[df['Station_Id_C']==56065]
df = df[df['Crop_LiStoc_Name']==2020028]

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


