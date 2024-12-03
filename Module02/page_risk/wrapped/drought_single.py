import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Module02.page_risk.wrapped.mci import calc_mci


def drought_cmip_single(cmip_data_dict, czt_data, yz_data, gdp_data):
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
            x['num'] = np.arange(len(x))
            x.dropna(how='any', inplace=True)
            train_x = x.iloc[:, -1].values.reshape(-1, 1)
            train_y = x.iloc[:, 0].values.reshape(-1, 1)
            model = LinearRegression(fit_intercept=True).fit(train_x, train_y)
            weight = model.coef_[0][0].round(3) * 10
            return weight
        except:
            return np.nan
    
    czt_val = czt_data.value.data  # 承灾体插值到站点后的静态值
    yz_val = yz_data.value.data  # 承灾体插值到站点后的静态值
    gdp_val = gdp_data.value.data

    risk_dict = dict()
    for exp, sub_dict in cmip_data_dict.items():
        risk_dict[exp] = dict()
        for insti, sub_dict1 in sub_dict.items():             
            # 读取数据转化为numpy array
            tem = sub_dict1['tas']
            tem_array = tem.tas.data
            tem_df = pd.DataFrame(tem_array, columns=tem.location, index=tem.time)
            tem_df = tem_df.resample('1M').mean()

            pre = sub_dict1['pr']
            pre_array = pre.pr.data
            pre_df = pd.DataFrame(pre_array, columns=pre.location, index=pre.time)
            pre_df = pre_df.resample('1M').sum()

            result_risk = []
            for i in range(len(czt_val)):
                col = tem_df.columns[i]
                        
                # 站点的危险性
                tmp_df = pd.concat([tem_df[col],pre_df[col]],axis=1)
                tmp_df.columns = ['TEM_Avg','PRE_Time_2020']
                mci = calc_mci(tmp_df, 0.3, 0.5, 0.3, 0.2)
                mci = mci[['轻度干旱', '中度干旱', '重度干旱', '特度干旱']]
                mci_year = mci.resample('1A').sum()
                mci_risk = 0.12*mci_year['轻度干旱'] + 0.23*mci_year['中度干旱'] + 0.37*mci_year['重度干旱'] + 0.28*mci_year['特度干旱']
                
                # 站点的承灾体和孕灾
                czt_risk = czt_val[i]
                yz_risk = yz_val[i]
                gdp_risk = gdp_val[i]
                
                # 最终风险
                total_risk = (0.42*mci_risk + 0.21*yz_risk + 0.25*czt_risk + 0.12*gdp_risk).round(3)
                result_risk.append(total_risk)
            
            result_risk = pd.concat(result_risk,axis=1)
            result_risk.columns = tem_df.columns
            result_risk.index = result_risk.index.strftime('%Y')

            # 创建临时下方统计的df
            tmp_df = pd.DataFrame(columns=result_risk.columns)
            tmp_df.loc['平均'] = result_risk.iloc[:, :].mean(axis=0).round(3)
            tmp_df.loc['变率'] = result_risk.apply(trend_rate, axis=0).round(3)
            tmp_df.loc['最大值'] = result_risk.iloc[:, :].max(axis=0).round(3)
            tmp_df.loc['最小值'] = result_risk.iloc[:, :].min(axis=0).round(3)

            # 合并所有结果
            stats_result = result_risk.copy()
            stats_result['区域均值'] = stats_result.iloc[:, :].mean(axis=1).round(3)
            stats_result['区域最大值'] = stats_result.iloc[:, :-3].max(axis=1).round(3)
            stats_result['区域最小值'] = stats_result.iloc[:, :-4].min(axis=1).round(3)
            stats_result = stats_result.round(3)

            # concat
            stats_result = pd.concat((stats_result, tmp_df), axis=0)

            # index处理
            stats_result.insert(loc=0, column='时间', value=stats_result.index)
            stats_result.reset_index(drop=True, inplace=True)
            
            risk_dict[exp][insti] = stats_result

    return risk_dict
