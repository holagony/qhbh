# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:12:31 2024

@author: EDY

EEMD方法
"""

from PyEMD import EMD, EEMD
from PyEMD.visualisation import Visualisation
import numpy as np
import pylab as plt
import pandas as pd
import warnings
import os
import matplotlib
from Module01.wrapped.func01_table_stats import table_stats
from Utils.data_processing import data_processing
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.config import cfg


matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def eemd(df, output_filepath):
    df_new = df.copy()
    df_new['区域平均'] = df_new.iloc[:, :].mean(axis=1).round(1)
    df_new['区域最大'] = df_new.iloc[:, :].max(axis=1)
    df_new['区域最小'] = df_new.iloc[:, :].min(axis=1)

    columns = df_new.columns.tolist()
    year = df_new.index.tolist()
    year = [int(y) for y in year]

    all_result = edict()
    for i in range(len(columns)):
        dat = df_new.iloc[:, i].values
        col = columns[i]
        name = ''.join(col)

        if np.any(np.isnan(dat)):
            # print(f'{columns[i]}存在nan值，时间序列不完整')
            all_result[name] = '该站点的时间序列不完整，不能生成结果'
            continue

        t0 = year[0]  # 开始的时间，以年为单位
        dt = 1  # 采样间隔，以年为单位
        N = dat.size  # 时间序列的长度
        t = np.arange(0, N) * dt + t0  # 构造时间序列数组

        p = np.polyfit(t - t0, dat, 1)  # 线性拟合
        dat_notrend = dat - np.polyval(p, t - t0)  # 去趋势
        std = dat_notrend.std()  # 标准差
        var = std**2  # 方差
        dat_norm = dat_notrend / std  # 标准化
        eemd = EEMD()
        emd = eemd.EMD
        eIMFs = eemd.eemd(dat_norm, t)
        nIMFs = eIMFs.shape[0]

        # Plot results
        fig = plt.figure(figsize=(8, 6))
        plt.subplot(nIMFs + 1, 1, 1)
        plt.plot(t, dat_norm, 'r')
        plt.title(columns[i])

        for n in range(nIMFs):
            plt.subplot(nIMFs + 1, 1, n + 2)
            plt.plot(t, eIMFs[n], 'g')
            plt.ylabel("eIMF %i" % (n + 1))
            plt.locator_params(axis='y', nbins=5)

        plt.xlabel("年份")
        plt.tight_layout()

        result_picture = os.path.join(output_filepath, name+'_eemd.png')
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()

        result_picture = result_picture.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
        result_picture = result_picture.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
        all_result[name] = result_picture

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

    # eemd
    save_file = r'C:/Users/MJY/Desktop/result'
    all_result = eemd(post_data_df, save_file)
