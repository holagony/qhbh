import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def drought_cmip_single(data, czt_data, yz_data, gdp_data):
    '''
    1.计算MCI
    2.计算危险性
    3.结合承灾体/孕灾环境/GPD，计算干旱风险
    多模式
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
            weight = model.coef_[0][0]*10
            return weight
        except:
            return np.nan
    
    czt_val = czt_data.value.data  # 承灾体插值到站点后的静态值
    yz_val = yz_data.value.data  # 承灾体插值到站点后的静态值
    gdp_val = gdp_data.value.data

    light_df = data['light_drought']
    light_df = light_df.resample('1A').sum()
    medium_df = data['medium_drought']
    medium_df = medium_df.resample('1A').sum()
    heavy_df = data['heavy_drought']
    heavy_df = heavy_df.resample('1A').sum()
    severe_df = data['severe_drought']
    severe_df = severe_df.resample('1A').sum()
        
    result_risk = []
    for i in range(len(czt_val)):
        col = light_df.columns[i]
        mci_year = pd.concat([light_df[col],medium_df[col],heavy_df[col],severe_df[col]],axis=1)
        mci_year.columns = ['轻度干旱', '中度干旱', '重度干旱', '特度干旱']
        mci_year = mci_year.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
        mci_year = mci_year.fillna(0)
        mci_risk = 0.12*mci_year['轻度干旱'] + 0.23*mci_year['中度干旱'] + 0.37*mci_year['重度干旱'] + 0.28*mci_year['特度干旱']
        
        # 站点的承灾体和孕灾
        czt_risk = czt_val[i]
        yz_risk = yz_val[i]
        gdp_risk = gdp_val[i]
        
        # 最终风险
        total_risk = 0.42*mci_risk + 0.21*yz_risk + 0.25*czt_risk + 0.12*gdp_risk
        result_risk.append(total_risk)
    
    result_risk = pd.concat(result_risk,axis=1)
    result_risk.columns = light_df.columns
    result_risk.index = result_risk.index.strftime('%Y')
    result_risk = result_risk.round(3)

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=result_risk.columns)
    tmp_df.loc['平均'] = result_risk.iloc[:, :].mean(axis=0).round(3)
    tmp_df.loc['变率'] = result_risk.apply(trend_rate, axis=0).round(6)
    tmp_df.loc['最大值'] = result_risk.iloc[:, :].max(axis=0).round(3)
    tmp_df.loc['最小值'] = result_risk.iloc[:, :].min(axis=0).round(3)

    # 合并所有结果
    stats_result = result_risk.copy()
    stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).round(3)
    stats_result['区域最大值'] = stats_result.iloc[:, :].max(axis=1).round(3)
    stats_result['区域最小值'] = stats_result.iloc[:, :].min(axis=1).round(3)

    # concat
    stats_result = pd.concat((stats_result, tmp_df), axis=0)

    for col in stats_result.columns:
        stats_result[col] = stats_result[col].astype(str).astype(float).round(3)

    # index处理
    stats_result.insert(loc=0, column='时间', value=stats_result.index)
    stats_result.reset_index(drop=True, inplace=True)
    
    return stats_result
