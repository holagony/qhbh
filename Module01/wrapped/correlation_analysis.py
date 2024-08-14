# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:09:14 2024

@author: EDY
"""

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


# matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


    
# path = r'D:\Project\3_项目\2_气候评估和气候可行性论证\qhkxxlz\Files\test_data\qh_mon.csv'
# data_dir = r'D:\Project\1'
# df = pd.read_csv(path, low_memory=False)
# df = data_processing(df)
# data_df = df[df.index.year <= 5000]
# refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
# nearly_df = df[df.index.year > 2011]
# last_year = 2023
# time_freq = 'M1'
# ele = 'TEM_Avg'
# stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)


def correlation_analysis(post_data_df,output_filepath):

    result=dict()
    df_sta_1=post_data_df.T.reset_index().T
    df_sta_1.columns =df_sta_1.iloc[0]
    df_sta_1 = df_sta_1.drop(df_sta_1.index[0])
    df_sta_1 = df_sta_1.drop(df_sta_1.index[0])

    columnsz=df_sta_1.columns
    for columns1 in columnsz:
        # print(columns1)
        r, q, p = sm.tsa.acf(post_data_df[columns1], nlags=20, fft=True, qstat=True) # alpha=0.05
        data = np.c_[range(1,21), r[1:], q, p]
        table = pd.DataFrame(data, columns=['Lag', "AC", "Q", "Prob(>Q)"])
        
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        fig = sm.graphics.tsa.plot_acf(post_data_df[columns1], lags=20, ax=ax1)
        plt.xlabel('滞后阶数')
        plt.ylabel('相关系数')
        plt.title(columns1)
    
        result_picture = os.path.join(output_filepath,'自相关_'+columns1+'.png')
        result[columns1]=result_picture
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.close()
            
        
        # 偏自相关
        r = sm.tsa.pacf(post_data_df[columns1], nlags=20)
        table = pd.DataFrame(r[1:], columns=['Lag'])
        
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        fig = sm.graphics.tsa.plot_pacf(post_data_df[columns1], lags=20, ax=ax1)
        plt.xlabel('滞后阶数')
        plt.ylabel('相关系数')
        plt.title(columns1)

        result_picture = os.path.join(output_filepath,'偏自相关_'+columns1+'.png')
        result[columns1]=result_picture
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.close()
        
    return result
    
if __name__ == "__main__":
    
    output_filepath=r'D:\Project\1'
    path = r'D:\Project\3_项目\2_气候评估和气候可行性论证\qhkxxlz\Files\test_data\qh_mon.csv'

    df = pd.read_csv(path, low_memory=False)
    df = data_processing(df)
    data_df = df[df.index.year<=5000]
    refer_df = df[(df.index.year>2000) & (df.index.year<2020)]
    nearly_df = df[df.index.year>2011]
    last_year = 2023
    time_freq = 'M1'
    ele = 'TEM_Avg'
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)

    result= correlation_analysis(post_data_df,output_filepath)
    
    

