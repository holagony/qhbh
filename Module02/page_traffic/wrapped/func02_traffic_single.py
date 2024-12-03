import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing


def traffic_cmip_single(cmip_data_dict, stats_result_his):
    '''
    计算集合的交通不利日数
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
    
    result = dict()
    for exp, sub_dict in cmip_data_dict.items():
        result[exp] = dict()
        for insti, sub_dict1 in sub_dict.items():             
            # 读取数据转化为numpy array
            tem = sub_dict1['tas']
            pre = sub_dict1['pr']
            u = sub_dict1['uas']
            v = sub_dict1['vas']
            wind = np.sqrt(u.uas**2 + v.vas**2)
            pre_array = pre.pr.data
            tem_array = tem.tas.data # - 273.15
            wind_array = wind.values
            
            # 根据规则判断是不是不利天气
            pre_flag = np.where((pre_array<50) | (np.isnan(pre_array)), 0, 1)
            tem_flag = np.where(((tem_array>0) & (tem_array<35)) | (np.isnan(tem_array)), 0, 1)
            wind_flag = np.where((wind_array<15) | (np.isnan(wind_array)), 0, 1)
            traffic_array = np.concatenate((pre_flag[None],tem_flag[None],wind_flag[None]),axis=0)
            traffic_array = np.max(traffic_array, axis=0)
            
            # 生成结果df
            traffic_cmip = pd.DataFrame(traffic_array, index=tem.time, columns=tem.location)
            traffic_cmip = traffic_cmip.resample('1A').sum()
            traffic_cmip.index = traffic_cmip.index.strftime('%Y')
            # traffic_cmip.reset_index(drop=False,inplace=True)
            # traffic_cmip.rename(columns={'index': '年份'}, inplace=True)
            
            # 创建临时下方统计的df
            tmp_df = pd.DataFrame(columns=traffic_cmip.columns)
            tmp_df.loc['平均'] = traffic_cmip.iloc[:, :].mean(axis=0).round(1)
            tmp_df.loc['变率'] = traffic_cmip.apply(trend_rate, axis=0).round(5)
            tmp_df.loc['最大值'] = traffic_cmip.iloc[:, :].max(axis=0).round(1)
            tmp_df.loc['最小值'] = traffic_cmip.iloc[:, :].min(axis=0).round(1)
            tmp_df.loc['参考时段均值'] = stats_result_his.iloc[-4, 1:-3]
            tmp_df.loc['距平'] = (tmp_df.loc['平均'].astype('float') - stats_result_his.iloc[-4,1:-3].astype('float')).round(1)
            tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'].astype('float') / stats_result_his.iloc[-4,1:-3].astype('float')) * 100).round(2)
            
            # 合并所有结果
            stats_result = traffic_cmip.copy()
            stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).astype(float).round(1)
            stats_result['区域最大值'] = stats_result.iloc[:, :-1].max(axis=1).astype(float).round(1)
            stats_result['区域最小值'] = stats_result.iloc[:, :-2].min(axis=1).astype(float).round(1)
            stats_result['区域距平'] = (stats_result['区域均值'] - stats_result_his.iloc[:-4,1:-3].mean().mean()).round(1)
            stats_result['区域距平百分率'] = ((stats_result['区域距平'] / stats_result_his.iloc[:-4,1:-3].mean().mean()) * 100).round(1)
            stats_result = stats_result.round(1)
    
            # concat
            stats_result = pd.concat((stats_result, tmp_df), axis=0)
    
            # index处理
            stats_result.insert(loc=0, column='时间', value=stats_result.index)
            stats_result.reset_index(drop=True, inplace=True)
            
            result[exp][insti] = stats_result.to_dict(orient='records')
    
    return result
