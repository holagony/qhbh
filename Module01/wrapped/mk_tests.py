import functools
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.data_processing import data_processing
from Module01.wrapped.table_stats import table_stats


def mann_kendall_mutation_test(df):
    '''
    MK突变点检验
    https://mp.weixin.qq.com/s/heonnS2lfEQSPz3Cnc0zvg
    https://mp.weixin.qq.com/s/GmUckn3SKfAmoXCVeJ7Iaw
    '''
    in_seq = df.iloc[:, 1].values
    n = in_seq.shape[0]
    NUMI = np.arange(2, n)
    E = NUMI * (NUMI - 1) / 4
    VAR = NUMI * (NUMI - 1) * (2 * NUMI + 5) / 72
    Ri = [(in_seq[i] > in_seq[1:i]).sum() for i in NUMI]
    Sk = np.cumsum(Ri)
    UFk = np.pad((Sk - E) / np.sqrt(VAR), (2, 0))
    Bk = np.cumsum([(in_seq[i] > in_seq[i:-1]).sum() for i in -(NUMI + 1)])
    UBk = np.pad((-(Bk - E) / np.sqrt(VAR)), (2, 0))[::-1]

    # 找出交叉点 (突变位置)
    point_idx = []
    diff = UFk - UBk
    for k in range(1, n):
        if diff[k - 1] * diff[k] < 0:
            point_idx.append(k)

    mutation_year = df.loc[point_idx, '时间'].tolist()
    if len(mutation_year) == 0:
        mutation_year = None

    df_out = df.copy()
    df_out['UFk'] = UFk.round(5)
    df_out['UBk'] = UBk.round(5)
    df_out.drop(columns=df_out.columns[1], axis=1, inplace=True)  # 删除原始要素数据
    df_out = df_out.droplevel(level=1, axis=1)

    return df_out, mutation_year


# 画图
# path = r'C:/Users/MJY/Desktop/data.xlsx'
# df_mean = pd.read_excel(path,sheet_name='mean')
# df1 = df_mean[['iyear','祁连山区']]
# df1.columns = ['年份','要素值']
# df2 = df_mean[['iyear','阿尼玛卿']]
# df2.columns = ['年份','要素值']
# df3 = df_mean[['iyear','各拉丹东']]
# df3.columns = ['年份','要素值']

# df_max = pd.read_excel(path,sheet_name='max')
# df4 = df_max[['iyear','祁连山区']]
# df4.columns = ['年份','要素值']
# df5 = df_max[['iyear','阿尼玛卿']]
# df5.columns = ['年份','要素值']
# df6 = df_max[['iyear','各拉丹东']]
# df6.columns = ['年份','要素值']

# df_min = pd.read_excel(path,sheet_name='min')
# df7 = df_min[['iyear','祁连山区']]
# df7.columns = ['年份','要素值']
# df8 = df_min[['iyear','阿尼玛卿']]
# df8.columns = ['年份','要素值']
# df9 = df_min[['iyear','各拉丹东']]
# df9.columns = ['年份','要素值']

# df_out, mutation_year = mann_kendall_mutation_test(df3)

# # 画图
# plt.figure(figsize=(8, 6), dpi=600)
# plt.plot(range(62), df_out['UFk'],  label='UF', color='blue', marker='s',markersize=4)
# plt.plot(range(62), df_out['UBk'], label='UB', color='red', linestyle='--', marker='o',markersize=4)
# ax1 = plt.gca()
# ax1.set_ylabel('统计量',fontname='MicroSoft YaHei', fontsize=10)
# ax1.set_xlabel('年份',fontname='MicroSoft YaHei', fontsize=10)
# plt.xlim(-1,62)             # 设置x轴、y轴范围
# # plt.ylim(-3,5)

# # 添加辅助线
# x_lim = plt.xlim()
# # 添加显著水平线和y=0
# plt.plot(x_lim,[-1.96,-1.96],':',color='green',label='0.05显著性水平')
# plt.plot(x_lim, [0,0],'-',color='black')
# plt.plot(x_lim,[1.96,1.96],':',color='green')
# plt.xticks(list(range(0,62,3)),labels=df_out['年份'][::3], rotation=45)

# # 设置图例
# legend = plt.legend(bbox_to_anchor=(0.3, 0.2))
# legend.get_frame().set_facecolor('white')  # 设置背景颜色为白色
# legend.get_frame().set_edgecolor('black')  # 设置边框颜色为黑色
# for text in legend.get_texts():
#     text.set_fontsize(12)  # 设置字体大小
#     text.set_fontfamily('MicroSoft YaHei')  # 设置字体名称

# plt.savefig("C:/Users/MJY/Desktop/result/1.png", dpi=300, bbox_inches='tight')
# plt.show()


def time_analysis(df):
    '''
    批量计算MK结果
    不能对日数据计算
    '''
    df = df.copy()
    df['区域平均'] = df.iloc[:, :].mean(axis=1).round(1)
    df['区域最大'] = df.iloc[:, :].max(axis=1)
    df['区域最小'] = df.iloc[:, :].min(axis=1)

    all_result = edict()
    for col in df.columns:
        data_tmp = df[col].to_frame()
        data_tmp.insert(loc=0, column='时间', value=data_tmp.index)
        data_tmp.reset_index(drop=True, inplace=True)
        result_out, _ = mann_kendall_mutation_test(data_tmp)  # 调用计算
        name = ''.join(col)
        all_result[name] = result_out

    return all_result


if __name__ == '__main__':
    path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
    df = pd.read_csv(path, low_memory=False)
    df = data_processing(df)
    data_df = df[df.index.year <= 2011]
    refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
    nearly_df = df[df.index.year > 2011]
    last_year = 2023
    time_freq = 'M1'
    ele = 'PRS_Avg'
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)

    # mk检验结果
    all_result = time_analysis(post_data_df)
