import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.data_processing import data_processing
from Utils.config import cfg


def table_stats_simple(refer_cmip):
    
    # 读取站点对应面积的csv数据
    sta_area = pd.read_csv(cfg.FILES.STATION_AREA)
    sta_area['区站号'] = sta_area['区站号'].map(str)
    sta_area['区站号'] = sta_area['区站号'].map(str)
    sta_area = sta_area[['区站号','面积']]
    sta_area.columns = ['Station_Id_C','面积']

    result_dict = dict()
    for exp, sub_dict1 in refer_cmip.items():
        result_dict[exp] = dict()
        for insti, data in sub_dict1.items():
            pre_df = data['pr']
            pre_df = pre_df.resample('1A').sum()
            result = []
            
            for col in pre_df.columns:
                area = sta_area[sta_area['Station_Id_C']==col]['面积'].values[0]
                rain_source = pre_df[col] * area
                result.append(rain_source)

            result = pd.concat(result,axis=1)
            base_p = result.mean(axis=0).round(1)
            result_dict[exp][insti] = base_p

    return result_dict
