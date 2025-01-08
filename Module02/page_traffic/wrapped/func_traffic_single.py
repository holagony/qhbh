import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def traffic_cmip_single(data, base_p):
    '''
    计算集合的交通不利日数
    '''
    
    def trend_rate(x):
        '''
        计算变率（气候倾向率）的pandas apply func
        '''
        try:
            x = x.to_frame()
            x['num'] = x.index.tolist()
            x['num'] = x['num'].map(int)
            x.dropna(how='any', inplace=True)
            train_x = x.iloc[:, -1].values.reshape(-1, 1)
            train_y = x.iloc[:, 0].values.reshape(-1, 1)
            model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
            weight = model.coef_[0][0] * 10
            return weight
        except:
            return np.nan

    # 根据规则判断是不是不利天气
    tem = data['tas']
    pre = data['pr']
    wind = data['ws']
    pre_flag = np.where((pre<50) | (np.isnan(pre)), 0, 1)
    tem_flag = np.where(((tem>0) & (tem<35)) | (np.isnan(tem)), 0, 1)
    wind_flag = np.where((wind<15) | (np.isnan(wind)), 0, 1)
    traffic_array = np.concatenate((pre_flag[None],tem_flag[None],wind_flag[None]),axis=0)
    traffic_array = np.max(traffic_array, axis=0)
    
    # 生成结果df
    traffic_cmip = pd.DataFrame(traffic_array, index=tem.index, columns=tem.columns)
    traffic_cmip = traffic_cmip.resample('1A').sum()
    traffic_cmip.index = traffic_cmip.index.strftime('%Y')

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=traffic_cmip.columns)
    tmp_df.loc['平均'] = traffic_cmip.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = traffic_cmip.apply(trend_rate, axis=0).round(3)
    tmp_df.loc['最大值'] = traffic_cmip.iloc[:, :].max(axis=0).round(1)
    tmp_df.loc['最小值'] = traffic_cmip.iloc[:, :].min(axis=0).round(1)
    tmp_df.loc['参考时段均值'] = base_p.round(1)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'].astype('float') - base_p).round(1)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'].astype('float') / base_p) * 100).round(1)

    # 合并所有结果
    stats_result = traffic_cmip.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).astype(float).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:, :-1].max(axis=1).astype(float).round(1)
    stats_result['区域最小值'] = stats_result.iloc[:, :-2].min(axis=1).astype(float).round(1)
    stats_result['区域距平'] = (stats_result['区域均值'] - base_p.mean()).round(1)
    stats_result['区域距平百分率'] = ((stats_result['区域距平'] / base_p.mean()) * 100).round(1)
    stats_result = stats_result.round(1)

    # concat
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    stats_result.reset_index(drop=True, inplace=True)
    
    return stats_result
