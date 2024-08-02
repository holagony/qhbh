import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing


def table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year):
    '''
    data_df 天擎统计时段数据
    refer_df 天擎参考时段数据
    nearly_df 天擎近10年数据
    time_freq 数据的时间类型 年/月/季/小时
    ele 计算的要素
    last_year 近1年年份
    '''
    last_df = nearly_df[nearly_df.index.year==last_year]
    last_df = last_df.pivot_table(index=last_df.index, columns=['Station_Name', 'Station_Id_C'], values=ele) # 近1年df
    data_df = data_df.pivot_table(index=data_df.index, columns=['Station_Name', 'Station_Id_C'], values=ele) # 统计时段df
    refer_df = refer_df.pivot_table(index=refer_df.index, columns=['Station_Name', 'Station_Id_C'], values=ele) # 参考时段df
    nearly_df = nearly_df.pivot_table(index=nearly_df.index, columns=['Station_Name', 'Station_Id_C'], values=ele) # 近10年df

    if time_freq in ['Y','Q']:
        data_df.index = data_df.index.strftime('%Y')
        refer_df.index = refer_df.index.strftime('%Y')
        nearly_df.index = nearly_df.index.strftime('%Y')
        last_df.index = last_df.index.strftime('%Y')

    elif time_freq in ['M1','M2']:
        data_df.index = data_df.index.strftime('%Y-%m')
        refer_df.index = refer_df.index.strftime('%Y-%m')
        nearly_df.index = nearly_df.index.strftime('%Y-%m')
        last_df.index = last_df.index.strftime('%Y-%m')

    elif time_freq in ['D1','D2']:
        data_df.index = data_df.index.strftime('%Y-%m-%d')
        refer_df.index = refer_df.index.strftime('%Y-%m-%d')
        nearly_df.index = nearly_df.index.strftime('%Y-%m-%d')
        last_df.index = last_df.index.strftime('%Y-%m-%d')

    def trend_rate(x):
        '''
        计算变率（气候倾向率）的pandas apply func
        '''
        try:
            x = x.to_frame()
            x['num'] = np.arange(len(x))
            x.dropna(how='any',inplace=True)
            train_x = x.iloc[:,-1].values.reshape(-1,1)
            train_y = x.iloc[:,0].values.reshape(-1,1)
            model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
            weight = model.coef_[0][0].round(3) * 10
            return weight
        except:
            return np.nan
        
    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df.columns)
    tmp_df.loc['平均'] = data_df.iloc[:,:].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df.apply(trend_rate,axis=0)
    tmp_df.loc['最大值'] = data_df.iloc[:,:].max(axis=0)
    tmp_df.loc['最小值'] = data_df.iloc[:,:].min(axis=0)
    tmp_df.loc['与上一年比较值'] = (data_df.iloc[:,:].mean(axis=0) - last_df.iloc[:,:].mean(axis=0)).round(1)
    tmp_df.loc['近10年均值'] = nearly_df.iloc[:,:].mean(axis=0).round(1)
    tmp_df.loc['与近10年比较值'] = (data_df.iloc[:,:].mean(axis=0) - nearly_df.iloc[:,:].mean(axis=0)).round(1)
    tmp_df.loc['参考时段均值'] = refer_df.iloc[:,:].mean(axis=0).round(1)
    tmp_df.loc['距平'] = tmp_df.loc['平均'] - tmp_df.loc['参考时段均值']
    tmp_df.loc['距平百分率%'] = ((tmp_df.loc['距平']/tmp_df.loc['参考时段均值'])*100).round(2)

    # 合并所有结果
    stats_result = data_df.copy()
    stats_result['区域均值'] = stats_result.iloc[:,:].mean(axis=1).round(1)
    stats_result['区域参考时段'] = np.nan
    stats_result['区域距平'] = (stats_result.iloc[:,:].mean(axis=1) - tmp_df.loc['参考时段均值'].mean()).round(1)
    stats_result['区域最大值'] = stats_result.iloc[:,:].max(axis=1)
    stats_result['区域最小值'] = stats_result.iloc[:,:].min(axis=1)
    stats_result = pd.concat((stats_result,tmp_df),axis=0)

    # index处理
    if time_freq in ['Y','Q']:
        stats_result.insert(loc=0, column='时间', value=stats_result.index)
    elif time_freq in ['M1','M2']:
        stats_result.insert(loc=0, column='时间', value=stats_result.index)
    elif time_freq in ['D1','D2']:
        stats_result.insert(loc=0, column='时间', value=stats_result.index)
        
    stats_result.reset_index(drop=True, inplace=True)
    post_data_df = data_df.copy()
    post_refer_df = refer_df.copy()
    
    return stats_result, post_data_df, post_refer_df


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
