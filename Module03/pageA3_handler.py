# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os


#%% main
def data_deal():

    #%% 文件保存位置
    #folder_path =r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\fake_data'
    folder_path ='/zipdata/fake_data'
    
    files_dict = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path)
            df = df.rename(columns={df.columns[0]: '时间'})
            df['时间'][1:] = df['时间'][1:].apply(lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else x)
            df = df.dropna(axis=1, how='all')
            for column in df.columns[1:]:  # 从第二列开始
                # 应用 lambda 函数，只对浮点数类型的数据保留两位小数
                df[column] = df[column].apply(lambda x: round(x, 2) if isinstance(x, (float, int)) else x)
            files_dict[os.path.splitext(filename)[0]] = df.to_dict(orient='records')
    
    
    return files_dict
        
if __name__ == '__main__':
    
    result_df_dict=data_deal()
    
