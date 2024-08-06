import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Utils.data_processing import data_processing
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
from Module01.wrapped.table_stats import table_stats
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm


# matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def linear_regression(x, y, intercept=1):

    if intercept == 1:
        flag = True
    else:
        flag = False

    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression(fit_intercept=flag).fit(x, y)
    weight = model.coef_[0][0].round(3)
    r_square = model.score(x, y)
    r_square = round(r_square, 3)

    if flag == True:
        bias = model.intercept_[0].round(3)
        return weight, bias, r_square
    elif flag == False:
        return weight, 0, r_square


def correlation_analysis(data_df, elements, main_st, sub_st, method, data_dir):
    '''
    相关性分析接口
    主站和对比站都是天擎站
    '''
    all_result = edict()
    all_result['picture'] = edict()
    all_result['picture'] = edict()
    all_result['data'] = edict()

    all_result['regression'] = edict()
    day_result_reg = pd.DataFrame(columns=['气象要素', '对比站X', '参证站Y', '样本数', '回归方程', 'Weight', 'Bias', 'R_square', 'X均值', 'Y均值'])

    # 计算
    x_train = data_df[main_st].to_frame()
    for sub in tqdm(sub_st):
        
        y_train = data_df[sub].to_frame()
        train = pd.concat([x_train, y_train], axis=1)
        train = train.dropna(how='any', axis=0)  # 删除任何包含nan的行

        train_data = train.values

        weight, bias, r_square = linear_regression(train_data[:, 0], train_data[:, 1])  # 计算线性回归
        formula = 'y = ' + str(weight) + 'x + ' + str(bias)
        num_data = len(train_data)
        df_row = day_result_reg.shape[0]
        day_result_reg.loc[df_row] = [elements, main_st, sub, num_data, formula, weight, bias, r_square, round(float(x_train.mean()),1), round(float(y_train.mean()),1)]
        

        # 图片绘制
        xy = train.T
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()

        z=(z-z.min())/(z.max()-z.min())
        fig, ax = plt.subplots(figsize=(7,5))
        scatter = ax.scatter(xy.iloc[0,:], xy.iloc[1,:], marker='o', c=z, edgecolors=None, s=15, cmap='RdBu_r',  alpha=0.8)
        cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30, label='frequency')

        regression_data=weight*xy.values.flatten()+bias
        plt.plot(xy.values.flatten(), regression_data, 'black', lw=1.5, label=f"回归方程: y={weight}x+{bias}") 
        
        x_line = np.linspace(xy.min().min(), xy.max().max(), 100)
        y_line = x_line
        plt.plot(x_line, y_line, c='red',linestyle='--')

        plt.xlim(xy.min().min(),xy.max().max())
        plt.ylim(xy.min().min(),xy.max().max())
        plt.xlabel(main_st[0])
        plt.ylabel(sub[0])
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best', frameon = False)

        picture=os.path.join(data_dir,f'{str(main_st)}_{str(sub)}_{ele}.png')
        plt.savefig(picture, bbox_inches='tight', dpi=200)
        plt.cla()
        plt.close('all')
        
        all_result['picture'][sub[1]]=picture
        # except:
        #     all_result['day'][ele]['data'][main_st[j]] = None

    # 保存计算结果
    if 'regression' in method:
        day_result_reg = day_result_reg
        all_result['regression'] = day_result_reg# .to_dict(orient='records')

        
    return all_result


if __name__ == '__main__':
    
    path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
    df = pd.read_csv(path, low_memory=False)
    df = data_processing(df)
    data_df = df[df.index.year <= 5000]
    refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
    nearly_df = df[df.index.year > 2011]
    last_year = 2023
    time_freq = 'M1'
    ele = 'TEM_Avg'
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, time_freq, ele, last_year)

    main_sta_ids = post_data_df.columns[0]
    sub_sta_ids = post_data_df.columns[1:].tolist()
    data_dir = r'C:/Users/MJY/Desktop/result'
    
    
    # https://www.cda.cn/discuss/post/details/5fc6f5fe20e51c68de404156
    r, q, p = sm.tsa.acf(post_data_df.iloc[:,0], nlags=20, fft=True, qstat=True) # alpha=0.05
    data = np.c_[range(1,21), r[1:], q, p]
    table = pd.DataFrame(data, columns=['Lag', "AC", "Q", "Prob(>Q)"])
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_acf(post_data_df.iloc[:,0], lags=20, ax=ax1)
    plt.xlabel('滞后阶数')
    plt.ylabel('相关系数')
    plt.show()
    

    # 偏自相关
    r = sm.tsa.pacf(post_data_df.iloc[:,0], nlags=20)
    table = pd.DataFrame(r[1:], columns=['Lag'])
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(post_data_df.iloc[:,0], lags=20, ax=ax1)
    plt.xlabel('滞后阶数')
    plt.ylabel('相关系数')
    plt.show()
    
    
    
    
    

