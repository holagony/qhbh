import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing


def rain_cmip_single(data, base_p, disaster, station_info):
    '''
    计算各情景单模式的气候变化风险预估-降水
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
    
    disaster_df = disaster.to_dataframe() # index是站号，columns=['disaster','lat','lon']
    station_info.set_index('站号', drop=False, inplace=True)
    pre_df = data['pr']

    rx5day_mon = pre_df.resample('1M').apply(lambda x: x.rolling(5).sum().max())
    rx5day_year = rx5day_mon.resample('1A').max()
    rx5day_year = rx5day_year.apply(lambda x: (x-x.min())/(x.max()-x.min()))
    rx5day_year.fillna(0, inplace=True)
    
    r20 = np.where(pre_df>=20, 1, 0)
    r20 = pd.DataFrame(r20, columns=pre_df.columns, index=pre_df.index)
    r20_year = r20.resample('1A').sum()
    r20_year = (r20_year-r20_year.min())/(r20_year.max()-r20_year.min()) # 0-1标准化
    r20_year.fillna(0, inplace=True)

    result_risk = []
    for col in pre_df.columns:
        risk = rx5day_year[col]*0.5 + r20_year[col]*0.4 + station_info.loc[col,'海拔']*0.1
        risk = (risk-risk.min())/(risk.max()-risk.min()) # 0-1标准化
        risk.fillna(0,inplace=True)
        risk.index = risk.index.strftime('%Y')
        risk = risk*disaster_df.loc[col,'disaster'] # 最后计算的风险值 0~1之间
        risk = risk.round(3)
        result_risk.append(risk)
    result_risk = pd.concat(result_risk,axis=1)
   
    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=result_risk.columns)
    tmp_df.loc['平均'] = result_risk.iloc[:, :].mean(axis=0).round(3)
    tmp_df.loc['变率'] = result_risk.apply(trend_rate, axis=0).round(3)
    tmp_df.loc['最大值'] = result_risk.iloc[:, :].max(axis=0).round(3)
    tmp_df.loc['最小值'] = result_risk.iloc[:, :].min(axis=0).round(3)
    tmp_df.loc['参考时段均值'] = base_p.round(3)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'].astype('float') - base_p).round(3)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'].astype('float') / base_p) * 100).round(1)
    
    # 合并所有结果
    stats_result = result_risk.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).round(3)
    stats_result['区域最大值'] = stats_result.iloc[:, :-1].max(axis=1).round(3)
    stats_result['区域最小值'] = stats_result.iloc[:, :-2].min(axis=1).round(3)
    stats_result['区域距平'] = (stats_result['区域均值'] - base_p.mean()).round(3)
    stats_result['区域距平百分率'] = ((stats_result['区域距平'] / base_p.mean()) * 100).round(1)

    # concat
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    stats_result.reset_index(drop=True, inplace=True)
                    
    return stats_result
