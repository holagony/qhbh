import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Utils.ordered_easydict import OrderedEasyDict as edict
from Module01.wrapped.func01_table_stats import table_stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

def calc_anomaly_cum(data_df, post_refer_df, save_file):
    '''
    计算累积距平
    '''
    all_result = dict()
    # all_result['img'] = dict()
    
    new_df = data_df.copy()
    new_df['区域平均'] = new_df.iloc[:, :].mean(axis=1).round(1)
    new_df['区域最大'] = new_df.iloc[:, :].max(axis=1)
    new_df['区域最小'] = new_df.iloc[:, :].min(axis=1)

    post_refer_df['区域平均'] = post_refer_df.iloc[:, :].mean(axis=1).round(1)
    post_refer_df['区域最大'] = post_refer_df.iloc[:, :].max(axis=1)
    post_refer_df['区域最小'] = post_refer_df.iloc[:, :].min(axis=1)
    refer_mean = post_refer_df.mean(axis=0).to_frame().T

    anomaly = []
    for col in new_df.columns:
        if col in refer_mean.columns:
            tmp = new_df[col] - refer_mean[col].values
            tmp = tmp.round(2)
            anomaly.append(tmp)

    anomaly = pd.concat(anomaly, axis=1) # 距平
    anomaly_accum = np.cumsum(anomaly, axis=0) # 累积距平
    anomaly_accum = anomaly_accum.round(2)
    
    # 画图
    # for col in anomaly.columns:
    #     bar_data = anomaly[col]
    #     line_data = anomaly_accum[col]
    #     fig, ax1 = plt.subplots(figsize=(8, 6))
    #     ax1.bar(range(len(anomaly)), bar_data, label='距平', color='blue')
    #     ax1.set_ylabel('距平')
    #     ax1.set_xlabel('年份', fontsize=10)
    #     ax1.axhline(0, color='black', linestyle='-')

    #     ax2 = ax1.twinx()
    #     ax2.plot(range(len(anomaly)), line_data, label='累积距平', color='red', linestyle='--', marker='o',markersize=4)
    #     ax2.set_ylabel('累积距平')
    #     plt.grid(ls="--", alpha=0.5)
    #     plt.xticks(list(range(0,len(anomaly),3)),labels=anomaly.index.tolist()[::3], rotation=45)
    #     fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    #     name = ''.join(col)
    #     save_path = os.path.join(save_file, name+'_距平.png')
    #     plt.savefig(save_path, dpi=200, bbox_inches='tight')
    #     plt.clf()
    #     plt.close()
    #     all_result['img'][name] = save_path
    
    # 保存
    anomaly.reset_index(drop=False,inplace=True)
    anomaly_accum.reset_index(drop=False,inplace=True)
    all_result['距平'] = anomaly.to_dict(orient='records')
    all_result['累积距平'] = anomaly_accum.to_dict(orient='records')

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

    # 累积距平
    save_file = r'C:/Users/MJY/Desktop/result'
    all_result = calc_anomaly_cum(post_data_df, post_refer_df, save_file)
