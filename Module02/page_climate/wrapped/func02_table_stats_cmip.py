import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing


def table_stats_simple_cmip(ds, stats_result_his, var, sta_list):
    data_df = pd.DataFrame(ds[var].data.astype(float).round(1))
    data_df.index = ds.time.dt.strftime('%Y')
    data_df.columns = sta_list
    
    def trend_rate(x):
        '''
        计算变率（气候倾向率）的pandas apply func
        '''
        try:
            x = x.to_frame()
            x['num'] = np.arange(len(x))
            x.dropna(how='any', inplace=True)
            train_x = x.iloc[:, -1].values.reshape(-1, 1)
            train_y = x.iloc[:, 0].values.reshape(-1, 1)
            model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
            weight = model.coef_[0][0].round(3) * 10
            return weight
        except:
            return np.nan

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df.iloc[:, :].mean(axis=0).astype(float).round(1)
    tmp_df.loc['变率'] = data_df.apply(trend_rate, axis=0).astype(float).round(5)
    tmp_df.loc['最大值'] = data_df.iloc[:, :].max(axis=0).astype(float).round(1)
    tmp_df.loc['最小值'] = data_df.iloc[:, :].min(axis=0).astype(float).round(1)
    
    if stats_result_his is not None:
        tmp_df.loc['参考时段均值'] = stats_result_his.iloc[-4, 1:-3]
        tmp_df.loc['距平'] = (tmp_df.loc['平均'].astype('float') - stats_result_his.iloc[-4,1:-3].astype('float')).round(1)
        tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'].astype('float') / stats_result_his.iloc[-4,1:-3].astype('float')) * 100).round(2)
    
    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).astype(float).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:, :-1].max(axis=1).astype(float).round(1)
    stats_result['区域最小值'] = stats_result.iloc[:, :-2].min(axis=1).astype(float).round(1)
    
    if stats_result_his is not None:
        stats_result['区域距平'] = (stats_result['区域均值'] - stats_result_his.iloc[:-4,1:-3].mean().mean()).round(1)
        stats_result['区域距平百分率'] = ((stats_result['区域距平'] / stats_result_his.iloc[:-4,1:-3].mean().mean()) * 100).round(1)
    
    stats_result = stats_result.astype(float).round(1)
    
    # concat
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    stats_result.reset_index(drop=True, inplace=True)
    
    return stats_result
