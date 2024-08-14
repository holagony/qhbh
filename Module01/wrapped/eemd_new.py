# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:32:50 2024

@author: EDY

EEMD方法

尝试使用多线程同时画图，失败
"""

from PyEMD import EEMD
import numpy  as np
import pylab as plt
import pandas as pd
import warnings
import os
import matplotlib
from multiprocessing import Pool, Manager
from matplotlib.path import Path
from Module01.wrapped.func01_table_stats import table_stats
from Utils.data_processing import data_processing

matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings('ignore')


def eemd_picture(result,i,df_sta_3,columns,year):
    dat=df_sta_3.iloc[:,i].values
    
    if np.any(np.isnan(dat)):
        print(f'{columns[i]}存在nan值，时间序列不完整')
    else:

        t0 = year[0]                               # 开始的时间，以年为单位
        dt = 1                                # 采样间隔，以年为单位
        N = dat.size                              # 时间序列的长度
        t = np.arange(0, N) * dt + t0             # 构造时间序列数组
        
        p = np.polyfit(t - t0, dat, 1)               # 线性拟合
        dat_notrend = dat - np.polyval(p, t - t0)    # 去趋势
        std = dat_notrend.std()                      # 标准差
        dat_norm = dat_notrend / std                 # 标准化
        
        eemd = EEMD()
         
         
        eIMFs = eemd.eemd(dat_norm, t)
        nIMFs = eIMFs.shape[0]
        
        # Plot results
        fig =plt.figure(figsize=(8,6))
        plt.subplot(nIMFs+1, 1, 1)
        plt.plot(t, dat_norm, 'r')
        plt.title(columns[i])
        
        for n in range(nIMFs):
            
            plt.subplot(nIMFs+1, 1, n+2)
            plt.plot(t, eIMFs[n], 'g')
            plt.ylabel("eIMF %i" %(n+1))
            plt.locator_params(axis='y', nbins=5)
        
        plt.xlabel("Year")
        plt.tight_layout()
        
        result_picture = os.path.join(output_filepath,'eemd_'+columns[i]+'.png')
        result[columns[i]]=result_picture
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.cla()
        
        result['eemd_'+columns[i]]=result_picture

        return result
    
def eemd(stats_result,output_filepath):
    df_sta_1=stats_result.T.reset_index()
    
    df_sta_1.columns =df_sta_1.iloc[0]
    df_sta_1 = df_sta_1.drop(df_sta_1.index[0])
    df_sta_1 = df_sta_1.iloc[:-5:,:]
    
    # 历年平均值
    df_sta_2 = df_sta_1.iloc[:,:-10:].T
    df_sta_2.columns =df_sta_2.iloc[0]
    df_sta_2 = df_sta_2.drop(df_sta_2.index[0])
    df_sta_2 = df_sta_2.drop(df_sta_2.index[0])
    
    df_sta_2.index = pd.DatetimeIndex(df_sta_2.index)
    df_sta_3 = df_sta_2.resample('Y').mean()
    year=df_sta_3.index.year
    columns=df_sta_3.columns
    
    manager = Manager()
    result = manager.dict()
    params_list = [(result,i,df_sta_3,columns,year) for i in range(len(columns))]

    # 使用进程池并行处理数据下载
    with Pool(processes=1) as pool:  # 假设我们使用4个进程
        pool.starmap(eemd_picture, params_list)
        
    return dict(result)

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

    result=eemd(stats_result,output_filepath)