import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Utils.ordered_easydict import OrderedEasyDict as edict
from Module01.wrapped.func01_table_stats import table_stats
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

def calc_moving_avg(data_df, window, save_file):
    '''
    计算滑动平均
    '''
    all_result = dict()
    all_result['img'] = dict()
    
    new_df = data_df.copy()
    new_df['区域平均'] = new_df.iloc[:, :].mean(axis=1).round(1)
    new_df['区域最大'] = new_df.iloc[:, :].max(axis=1)
    new_df['区域最小'] = new_df.iloc[:, :].min(axis=1)
    moving_result = new_df.apply(lambda x: x.rolling(window).mean().round(2))
    
    # 画图
    for col in moving_result.columns:
        origin_data = new_df[col]
        smooth_data = moving_result[col]
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.set_xlabel('年份',fontname='MicroSoft YaHei', fontsize=10)
        ax1.set_ylabel('要素值')
        ax1.plot(range(len(origin_data)), origin_data, label='origin data', color='orange', linestyle='-', marker='o',markersize=4)
        ax1.plot(range(len(smooth_data)), smooth_data, label='smoothed data', color='blue', linestyle='--')
        plt.grid(ls="--", alpha=0.5)
        plt.xticks(list(range(0,len(origin_data),2)),labels=origin_data.index.tolist()[::2], rotation=45)
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        name = ''.join(col)
        save_path = os.path.join(save_file, name+'_滑动平均.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
        all_result['img'][name] = save_path
    
    # 保存
    moving_result.reset_index(drop=False,inplace=True)
    all_result['滑动平均'] = moving_result.to_dict(orient='records')
    
    return all_result


if __name__ == '__main__':
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
    
    # 滑动平均
    save_file = r'C:/Users/MJY/Desktop/result'
    all_result = calc_moving_avg(post_data_df, 3, save_file)























    
    
    
    