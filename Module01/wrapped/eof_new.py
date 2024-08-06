import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.data_processing import data_processing
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
from Module01.wrapped.table_stats import table_stats
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import xarray as xr
import xeofs as xe

# https://blog.csdn.net/zxqxr/article/details/130650878

path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
df = pd.read_csv(path, low_memory=False)
df = data_processing(df)
data_df = df[df.index.year <= 5000]
refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
nearly_df = df[df.index.year > 2011]
last_year = 2023
time_freq = 'M1'
ele = 'TEM_Avg'
stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)

 
t2m = xr.tutorial.open_dataset("air_temperature")

eof = xe.models.EOF(n_modes=10)
eof.fit(t2m, dim="time")
comps = eof.components()  # EOFs (spatial patterns)
scores = eof.scores()  # PCs (temporal patterns)

rotator = xe.models.EOFRotator(n_modes=3)
rotator.fit(eof) # doctest: +ELLIPSIS
rot_comps = rotator.components()  # Rotated EOFs (spatial patterns)
rot_scores = rotator.scores()  # Rotated PCs (temporal patterns)

