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
    
    
def stats_result_1(df_in, refer_df):
    '''
    模拟(观测) 使用验证期的气象数据计算的HBV结果
    '''
    station_name = refer_df['Station_Name'][0]
    sta_id = refer_df['Station_Id_C'][0]
    df_in.index = df_in.index.strftime('%Y')
    
    # 横向的距平和距平百分率
    df = pd.DataFrame(index=df_in.index)
    df['距平'] = (df_in['Q'] - refer_df['Q'].mean(axis=0)).round(1)
    df['距平百分率'] = ((df['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(2)

    # 创建临时下方统计的df
    tmp_df = pd.DataFrame(columns=df_in.columns)
    tmp_df.loc['平均'] = df_in.iloc[:, :].mean(axis=0).round(1)
    tmp_df.loc['变率'] = df_in.apply(trend_rate, axis=0).round(1)
    tmp_df.loc['最大值'] = df_in.iloc[:, :].max(axis=0)
    tmp_df.loc['最小值'] = df_in.iloc[:, :].min(axis=0)
    tmp_df.loc['距平'] = (tmp_df.loc['平均'] - refer_df['Q'].mean(axis=0)).round(1)
    tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(2)
    tmp_df.loc['参考时段'] = refer_df['Q'].mean(axis=0).round(1)

    result = pd.concat([df_in,tmp_df],axis=0)
    result = pd.concat([result,df],axis=1)
    
    result['站名'] = station_name
    result['站号'] = sta_id
    result = result[['站名','站号','Q','距平','距平百分率']]
    result.reset_index(drop=False,inplace=True)
    result = result#.to_dict(orient='records')
    
    return result


def stats_result_2(dict_in, refer_df):
    '''
    模拟(模式) 使用验证期的模式数据计算径流 情境下集合平均
    预估 预估数据计算径流 情境下集合平均
    '''
    station_name = refer_df['Station_Name'][0]
    sta_id = refer_df['Station_Id_C'][0]
    
    for exp, df_in in dict_in.items():
        df_in.index = df_in.index.strftime('%Y')
        
        # 横向的距平和距平百分率
        df = pd.DataFrame(index=df_in.index)
        df['距平'] = df_in['Q'] - refer_df['Q'].mean(axis=0).round(1)
        df['距平百分率'] = ((df['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(1)
    
        # 创建临时下方统计的df
        tmp_df = pd.DataFrame(columns=df_in.columns)
        tmp_df.loc['平均'] = df_in.iloc[:, :].mean(axis=0).round(1)
        tmp_df.loc['变率'] = df_in.apply(trend_rate, axis=0).round(1)
        tmp_df.loc['最大值'] = df_in.iloc[:, :].max(axis=0)
        tmp_df.loc['最小值'] = df_in.iloc[:, :].min(axis=0)
        tmp_df.loc['距平'] = (tmp_df.loc['平均'] - refer_df['Q'].mean(axis=0)).round(1)
        tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(1)
        tmp_df.loc['参考时段'] = refer_df['Q'].mean(axis=0).round(1)

        result = pd.concat([df_in,tmp_df],axis=0)
        result = pd.concat([result,df],axis=1)
        
        result['站名'] = station_name
        result['站号'] = sta_id
        result = result[['站名','站号','Q','距平','距平百分率']]
        result.reset_index(drop=False,inplace=True)
        dict_in[exp] = result.to_dict(orient='records')

    return dict_in


def stats_result_3(dict_in, refer_df):
    '''
    预估 单模式，每个模式里面包含所有情景
    '''
    station_name = refer_df['Station_Name'][0]
    sta_id = refer_df['Station_Id_C'][0]
    
    for exp, sub_dict in dict_in.items():
        for insti, df_in in sub_dict.items():
            
            df_in.index = df_in.index.strftime('%Y')
            
            # 横向的距平和距平百分率
            df = pd.DataFrame(index=df_in.index)
            df['距平'] = (df_in['Q'] - refer_df['Q'].mean(axis=0)).round(1)
            df['距平百分率'] = ((df['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(1)
        
            # 创建临时下方统计的df
            tmp_df = pd.DataFrame(columns=df_in.columns)
            tmp_df.loc['平均'] = df_in.iloc[:, :].mean(axis=0).round(1)
            tmp_df.loc['变率'] = df_in.apply(trend_rate, axis=0).round(1)
            tmp_df.loc['最大值'] = df_in.iloc[:, :].max(axis=0)
            tmp_df.loc['最小值'] = df_in.iloc[:, :].min(axis=0)
            tmp_df.loc['距平'] = (tmp_df.loc['平均'] - refer_df['Q'].mean(axis=0)).round(1)
            tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'] / refer_df['Q'].mean(axis=0)) * 100).round(1)
            tmp_df.loc['参考时段'] = refer_df['Q'].mean(axis=0).round(1)
    
            result = pd.concat([df_in,tmp_df],axis=0)
            result = pd.concat([result,df],axis=1)
            
            result['站名'] = station_name
            result['站号'] = sta_id
            result = result[['站名','站号','Q','距平','距平百分率']]
            result.reset_index(drop=False,inplace=True)
            sub_dict[insti] = result.to_dict(orient='records')
    
    return dict_in


def stats_result_4(single_cmip_res, base_p, hydro_name, hydro_id):
    '''
    hbv改
    '''
    for exp, sub_dict in single_cmip_res.items():
        for insti, df_in in sub_dict.items():
            
            df_in.index = df_in.index.strftime('%Y')
            
            # 横向的距平和距平百分率
            df = pd.DataFrame(index=df_in.index)
            df['距平'] = (df_in['Q'] - base_p).round(1)
            df['距平百分率'] = ((df['距平'] / base_p) * 100).round(1)
        
            # 创建临时下方统计的df
            tmp_df = pd.DataFrame(columns=df_in.columns)
            tmp_df.loc['平均'] = df_in.iloc[:, :].mean(axis=0).round(1)
            tmp_df.loc['变率'] = df_in.apply(trend_rate, axis=0).round(1)
            tmp_df.loc['最大值'] = df_in.iloc[:, :].max(axis=0)
            tmp_df.loc['最小值'] = df_in.iloc[:, :].min(axis=0)
            tmp_df.loc['距平'] = (tmp_df.loc['平均'] - base_p).round(1)
            tmp_df.loc['距平百分率'] = ((tmp_df.loc['距平'] / base_p) * 100).round(1)
            tmp_df.loc['参考时段'] = base_p
    
            result = pd.concat([df_in,tmp_df],axis=0)
            result = pd.concat([result,df],axis=1)
            
            result['站名'] = hydro_name
            result['站号'] = hydro_id
            result = result[['站名','站号','Q','距平','距平百分率']]
            result.reset_index(drop=False,inplace=True)
            sub_dict[insti] = result
    
    return single_cmip_res

