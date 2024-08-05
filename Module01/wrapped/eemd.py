# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:12:31 2024

@author: EDY

EEMD方法
"""

from PyEMD import EMD, EEMD
from PyEMD.visualisation import Visualisation 
import numpy  as np
import pylab as plt
import pandas as pd
import warnings
import os
import matplotlib

matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings('ignore')


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
    
    result=dict()
    for i in range(len(columns)):
        dat=df_sta_3.iloc[:,i].values
        t0 = year[0]                               # 开始的时间，以年为单位
        dt = 1                                # 采样间隔，以年为单位
        N = dat.size                              # 时间序列的长度
        t = np.arange(0, N) * dt + t0             # 构造时间序列数组
        
        p = np.polyfit(t - t0, dat, 1)               # 线性拟合
        dat_notrend = dat - np.polyval(p, t - t0)    # 去趋势
        std = dat_notrend.std()                      # 标准差
        var = std ** 2                               # 方差
        dat_norm = dat_notrend / std                 # 标准化
        
        eemd = EEMD()
         
        emd = eemd.EMD
         
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
        
        result_picture = os.path.join(output_filepath,columns[i]+'.png')
        result[columns[i]]=result_picture
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.cla()
if __name__ == "__main__":
    
    output_filepath=r'D:\Project\1'
    result=eemd(stats_result,output_filepath)