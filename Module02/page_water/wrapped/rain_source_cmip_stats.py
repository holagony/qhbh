import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Utils.config import cfg


def table_stats_simple_cmip(data, base_p):

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
    
    # 读取站点对应面积的csv数据
    sta_area = pd.read_csv(cfg.FILES.STATION_AREA)
    sta_area['区站号'] = sta_area['区站号'].map(str)
    sta_area['区站号'] = sta_area['区站号'].map(str)
    sta_area = sta_area[['区站号','面积']]
    sta_area.columns = ['Station_Id_C','面积']
    
    data_df = data['pr']
    data_df_yearly = data_df.resample('1A').sum()
    data_df_yearly.index = data_df_yearly.index.year

    
    # 计算降水资源量
    for col in data_df.columns:
        area = float(sta_area[sta_area['Station_Id_C']==col]['面积'])
        data_df_yearly[col] = (data_df_yearly[col] * area).round(1)

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df_yearly.iloc[:, :].mean(axis=0).astype(float).round(1)
    tmp_df.loc['变率'] = data_df_yearly.apply(trend_rate, axis=0).astype(float).round(3)
    tmp_df.loc['最大值'] = data_df_yearly.iloc[:, :].max(axis=0).astype(float).round(1)
    tmp_df.loc['最小值'] = data_df_yearly.iloc[:, :].min(axis=0).astype(float).round(1)
    tmp_df.loc['参考时段均值'] = base_p.round(1)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'].astype('float') - base_p).round(1)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'].astype('float') / base_p) * 100).round(1)
    
    # 合并所有结果
    stats_result = data_df_yearly.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).astype(float).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:, :-1].max(axis=1).astype(float).round(1)
    stats_result['区域最小值'] = stats_result.iloc[:, :-2].min(axis=1).astype(float).round(1)
    stats_result['区域距平'] = (stats_result['区域均值'] - base_p.mean()).round(1)
    stats_result['区域距平百分率'] = ((stats_result['区域距平'] / base_p.mean()) * 100).round(1)
    # stats_result = stats_result.astype(float).round(1)
    
    # concat
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    stats_result.reset_index(drop=True, inplace=True)
    
    return stats_result
