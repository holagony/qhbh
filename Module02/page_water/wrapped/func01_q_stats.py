import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    
    
def stats_q(data_df, refer_df):
    '''
    根据验收期的时间选择，统计水文站实测数据的径流量
    '''
    station_name = data_df['Station_Name'][0]
    sta_id = data_df['Station_Id_C'][0]
    data_df_Q = data_df['Q'].resample('1A').mean().round(2).to_frame()
    data_df_Q.index = data_df_Q.index.strftime('%Y')
    
    # 横向的距平和距平百分率
    df = pd.DataFrame(index=data_df_Q.index)
    df['距平'] = data_df_Q['Q'] - refer_df['Q'].mean(axis=0)
    df['距平百分率'] = ((df['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(2)

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df_Q.columns)
    tmp_df.loc['平均'] = data_df_Q.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df_Q.apply(trend_rate, axis=0).round(1)
    tmp_df.loc['最大值'] = data_df_Q.iloc[:, :].max(axis=0)
    tmp_df.loc['最小值'] = data_df_Q.iloc[:, :].min(axis=0)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'] - refer_df['Q'].mean(axis=0)).round(1)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(2)
    tmp_df.loc['参考时段'] = refer_df['Q'].mean(axis=0).round(1)

    result = pd.concat([data_df_Q,tmp_df],axis=0)
    result = pd.concat([result,df],axis=1)
    
    result['站名'] = station_name
    result['站号'] = sta_id
    result = result[['站名','站号','Q','距平','距平百分率']]
    result.reset_index(drop=False,inplace=True)
    
    return result


if __name__ == '__main__':
    pass
