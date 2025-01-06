import os
import glob
import json
import numpy as np
import pandas as pd
from Module02.page_risk.wrapped.mci import calc_mci
from Utils.config import cfg


def rain_change_processing(refer_cmip, disaster, station_info):
    '''
    气候变化影响预估-降水-站点数据处理，计算Rx5day和R20mm
    转换到年尺度，计算风险，计算基准期
    '''
    disaster_df = disaster.to_dataframe() # index是站号，columns=['disaster','lat','lon']
    station_info.set_index('站号', drop=False, inplace=True)
    result_dict = dict()
    for exp, sub_dict1 in refer_cmip.items():
        result_dict[exp] = dict()
        for insti, data in sub_dict1.items():
            pre_df = data['pr']
            rx5day = pre_df.resample('1M').apply(lambda x: x.rolling(5).sum().max())
            rx5day = rx5day.resample('1A').max()
            rx5day = (rx5day-rx5day.min())/(rx5day.max()-rx5day.min()) # 0-1标准化
            rx5day.fillna(0, inplace=True)

            pre_day = np.where(pre_df>=20, 1, 0)
            pre_day = pd.DataFrame(pre_day,index=pre_df.index,columns=pre_df.columns)
            r20 = pre_day.resample('1A').sum()
            r20 = (r20-r20.min())/(r20.max()-r20.min()) # 0-1标准化
            r20.fillna(0, inplace=True)

            final_risk = []
            for col in pre_df.columns:
                risk = rx5day[col]*0.5 + r20[col]*0.4 + station_info.loc[col,'海拔']*0.1
                risk = (risk-risk.min())/(risk.max()-risk.min()) # 0-1标准化
                risk.fillna(0,inplace=True)
                risk = risk*disaster_df.loc[col,'disaster'] # 最后计算的风险值 0~1之间
                final_risk.append(risk)
                
            final_risk = pd.concat(final_risk,axis=1)
            base_p = final_risk.mean(axis=0).round(3) # 基准期
            result_dict[exp][insti] = base_p.round(3)

    return result_dict


def drought_change_processing(refer_cmip):
    '''
    气候变化影响预估-干旱-站点数据处理，计算MCI
    日数据转换成月结果，计算完之后转换到年尺度，计算风险性
    '''
    result_dict = dict()
    for exp, sub_dict1 in refer_cmip.items():
        result_dict[exp] = dict()
        for insti, data in sub_dict1.items():
            tas = data['tas'].resample('1M').mean()
            pr = data['pr'].resample('1M').sum()
            
            mci_res = []
            for st_id in tas.columns:
                concat = pd.concat([tas[st_id], pr[st_id]], axis=1)
                concat.columns = ['TEM_Avg', 'PRE_Time_2020']
                concat = calc_mci(concat, 0.3, 0.5, 0.3, 0.2)
                base_p = concat['干旱指数'].mean(axis=0)
                mci_res.append(base_p)
            mci_res = pd.concat(mci_res, axis=1)
            result_dict[exp][insti] = mci_res
    
    return result_dict
