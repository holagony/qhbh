import numpy as np
import pandas as pd


def refer_traffic_processing(refer_cmip):
    '''
    把参考时段读取的原始模式数据转换为交通不利日数，并计算基准期
    '''
    result_dict = dict()
    for exp, sub_dict1 in refer_cmip.items():
        result_dict[exp] = dict()
        for insti, data in sub_dict1.items():
            pre_flag = np.where((data['pr']<50) | (pd.isna(data['pr'])), 0, 1)
            tem_flag = np.where(((data['tas']>0) & (data['tas']<35)) | (pd.isna(data['tas'])),0,1)
            win_flag = np.where((data['ws']<15) | (pd.isna(data['ws'])),0,1)
            traffic = np.concatenate((pre_flag[None],tem_flag[None],win_flag[None]),axis=0)
            traffic = np.max(traffic, axis=0)
            traffic = pd.DataFrame(traffic,columns=data['pr'].columns,index=data['pr'].index)
            traffic_yearly = traffic.resample('1A').sum()
            base_p = traffic_yearly.mean(axis=0)
            result_dict[exp][insti] = base_p
            
    return result_dict