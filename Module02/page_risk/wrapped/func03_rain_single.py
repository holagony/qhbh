import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing


def rain_cmip_single(cmip_data_dict, stats_result_his, disaster, alti_list):
    '''
    计算各情景单模式的气候变化风险预估-降水
    '''
    
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
    
    alti_list = np.array(alti_list).reshape(1,-1)
    disaster_val = disaster.disaster.data.reshape(1,-1) # 承载体插值到站点后的静态值
    result = dict()
    for exp, sub_dict in cmip_data_dict.items():
        result[exp] = dict()
        for insti, sub_dict1 in sub_dict.items():             
            # 读取数据转化为numpy array
            
            pre = sub_dict1['pr']
            pre_array = pre.pr.data
            pre_df = pd.DataFrame(pre_array, columns=pre.location, index=pre.time)
            
            rx5day_mon = pre_df.resample('1M').apply(lambda x: x.rolling(5).sum().max())
            rx5day_year = rx5day_mon.resample('1A').max()
            rx5day_year = rx5day_year.apply(lambda x: (x-x.min())/(x.max()-x.min()))
            rx5day_year.fillna(0, inplace=True)
            
            r20 = np.where(pre_df>=20, 1, 0)
            r20 = pd.DataFrame(r20, columns=pre.location, index=pre.time)
            r20_year = r20.resample('1A').sum()
            r20_year = (r20_year-r20_year.min())/(r20_year.max()-r20_year.min()) # 0-1标准化
            r20_year.fillna(0, inplace=True)
        
            risk = 0.5*rx5day_year + 0.4*r20_year + 0.1*alti_list
            risk = (risk-risk.min())/(risk.max()-risk.min()) # 0-1标准化
            risk.fillna(0, inplace=True)
            risk.index = risk.index.strftime('%Y')
            
            result_risk = risk*disaster_val # 最后计算的风险值 0~1之间
            result_risk = result_risk.round(3)
            
            # 创建临时下方统计的df
            tmp_df = pd.DataFrame(columns=result_risk.columns)
            tmp_df.loc['平均'] = result_risk.iloc[:, :].mean(axis=0).round(3)
            tmp_df.loc['变率'] = result_risk.apply(trend_rate, axis=0).round(3)
            tmp_df.loc['最大值'] = result_risk.iloc[:, :].max(axis=0).round(3)
            tmp_df.loc['最小值'] = result_risk.iloc[:, :].min(axis=0).round(3)
            tmp_df.loc['参考时段均值'] = stats_result_his.iloc[-4, 1:-3]
            tmp_df.loc['距平'] = (tmp_df.loc['平均'].astype('float') - stats_result_his.iloc[-4,1:-3].astype('float')).round(3)
            tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'].astype('float') / stats_result_his.iloc[-4,1:-3].astype('float')) * 100).round(2)
            
            # 合并所有结果
            stats_result = result_risk.copy()
            stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).round(3)
            stats_result['区域最大值'] = stats_result.iloc[:, :-1].max(axis=1).round(3)
            stats_result['区域最小值'] = stats_result.iloc[:, :-2].min(axis=1).round(3)
            stats_result['区域距平'] = (stats_result['区域均值'] - stats_result_his.iloc[:-4,1:-3].mean().mean()).round(3)
            stats_result['区域距平百分率'] = ((stats_result['区域距平'] / stats_result_his.iloc[:-4,1:-3].mean().mean()) * 100).round(3)
            stats_result = stats_result.round(3)

            # concat
            stats_result = pd.concat((stats_result, tmp_df), axis=0)

            # index处理
            stats_result.insert(loc=0, column='时间', value=stats_result.index)
            stats_result.reset_index(drop=True, inplace=True)
                        
            result[exp][insti] = stats_result.to_dict(orient='records')
    
    return result
