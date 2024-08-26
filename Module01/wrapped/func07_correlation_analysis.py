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
from Module01.wrapped.func01_table_stats import table_stats
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def correlation_analysis(df, output_filepath):
    '''
    计算自相关和偏相关画图
    '''
    new_df = df.copy()
    new_df['区域平均'] = new_df.iloc[:, :].mean(axis=1).round(1)
    new_df['区域最大'] = new_df.iloc[:, :].max(axis=1)
    new_df['区域最小'] = new_df.iloc[:, :].min(axis=1)
    
    num = 10 # 滞后阶数
    columnsz = new_df.columns.tolist()
    all_result = edict()

    for columns1 in columnsz:
        name = ''.join(columns1)
        all_result[name] = edict()

        r, q, p = sm.tsa.acf(new_df[columns1], nlags=num, fft=True, qstat=True)  # alpha=0.05
        data = np.c_[range(1, num+1), r[1:], q, p]
        table = pd.DataFrame(data, columns=['Lag', "AC", "Q", "Prob(>Q)"])
        all_result[name]['自相关'] = table.to_dict(orient='records')

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        fig = sm.graphics.tsa.plot_acf(new_df[columns1], lags=num, ax=ax1)
        plt.xlabel('滞后阶数')
        plt.ylabel('相关系数')
        plt.title(columns1)

        result_picture = os.path.join(output_filepath, name + '_自相关.png')
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
        result_picture = result_picture.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
        result_picture = result_picture.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        all_result[name]['img'] = result_picture

        # 偏自相关
        r = sm.tsa.pacf(new_df[columns1], nlags=num)
        table = pd.DataFrame(r[1:], columns=['偏相关系数'])
        table.reset_index(drop=False,inplace=True)
        table.columns = ['Lags','偏相关系数']
        all_result[name]['偏自相关'] = table.to_dict(orient='records')

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        fig = sm.graphics.tsa.plot_pacf(new_df[columns1], lags=num, ax=ax1)
        plt.xlabel('滞后阶数')
        plt.ylabel('相关系数')
        plt.title(columns1)

        result_picture = os.path.join(output_filepath, name + '_偏自相关.png')
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()

        result_picture = result_picture.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
        result_picture = result_picture.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        all_result[name]['p_img'] = result_picture

    return all_result


if __name__ == "__main__":
    path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
    df = pd.read_csv(path, low_memory=False)
    element = 'TEM_Avg'
    df = df[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Datetime', 'Year', 'Mon', element]]
    df = data_processing(df, element)
    data_df = df[df.index.year <= 2011]
    refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
    nearly_df = df[df.index.year > 2011]
    last_year = 2023
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, element, last_year)

    # 自相关
    save_file = r'C:/Users/MJY/Desktop/result'
    all_result = correlation_analysis(post_data_df, save_file)

