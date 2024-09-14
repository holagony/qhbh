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

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=data_df_Q.columns)
    tmp_df.loc['平均'] = data_df_Q.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = data_df_Q.apply(trend_rate, axis=0).round(1)
    tmp_df.loc['最大值'] = data_df_Q.iloc[:, :].max(axis=0)
    tmp_df.loc['最小值'] = data_df_Q.iloc[:, :].min(axis=0)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'] - refer_df['Q'].mean(axis=0)).round(1)
    tmp_df.loc['距平百分率%'] = ((tmp_df.loc['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(2)
    tmp_df.loc['参考时段'] = refer_df['Q'].mean(axis=0).round(1)

    result = pd.concat([data_df_Q,tmp_df],axis=0)
    result['站名'] = station_name
    result['站号'] = sta_id
    result = result[['站名','站号','Q']]
    result.reset_index(drop=False,inplace=True)
    
    return result


if __name__ == '__main__':
    # from Module02.page_water.page_water_hbv import hbv_single_calc
    # data_json = dict()
    # data_json['time_freq'] = 'Y'
    # data_json['evaluate_times'] = '1950,1980' # 预估时段时间条
    # data_json['refer_years'] = '2023,2024'# 参考时段时间条
    # data_json['valid_times'] = '202303,202403' # 验证期 '%Y%m,%Y%m'
    # data_json['hydro_ids'] = '40100350' # 唐乃亥
    # data_json['sta_ids'] = '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    # data_json['cmip_type'] = 'original' # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    # data_json['cmip_res'] = None # 分辨率 1/5/10/25/50/100 km
    # data_json['cmip_model'] = ['BCC-CSM2-MR', 'CanESM5']# 模式，列表：['CanESM5','CESM2']等
    # data_json['degree'] = None
    # data_df, refer_df, data_df_meteo, vaild_cmip, evaluate_cmip = hbv_single_calc(data_json)

    # result = stats_q(data_df, refer_df)
    pass
