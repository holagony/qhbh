import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Module01.wrapped.table_stats import table_stats


def calc_anomaly_cum(df, refer_df):
    '''
    计算累积距平
    '''
    new_df = df.copy()
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
    
    anomaly = pd.concat(anomaly,axis=1)
    anomaly_accum = np.cumsum(anomaly,axis=0)
    anomaly_accum = anomaly_accum.round(2)
    
    return anomaly, anomaly_accum




if __name__ == '__main__':
    path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
    df = pd.read_csv(path, low_memory=False)
    df = data_processing(df)
    data_df = df[df.index.year<=2011]
    refer_df = df[(df.index.year>2000) & (df.index.year<2020)]
    nearly_df = df[df.index.year>2011]
    last_year = 2023
    time_freq = 'M1'
    ele = 'PRS_Avg'
    
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)
    anomaly, anomaly_accum = calc_anomaly_cum(post_data_df, post_refer_df)
































    
    
    
    