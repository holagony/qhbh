import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing


def table_stats_simple(data_df, ele,flag='his'):

    if flag=='his':
        data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Id_C'], values=ele)  # 统计时段df
        data_df.index = data_df.index.strftime('%Y')
    else:
        data_df.index = data_df.index.strftime('%Y')

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
    tmp_df.loc['变率'] = data_df.apply(trend_rate, axis=0).round(5)

    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).astype(float).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:, :].max(axis=1)
    stats_result['区域最小值'] = stats_result.iloc[:, :].min(axis=1)
    
    
    # concat
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    stats_result.reset_index(drop=True, inplace=True)

    
    return stats_result

