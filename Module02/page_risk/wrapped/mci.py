import numpy as np
import pandas as pd
from scipy.stats import gamma, norm


def calc_mi(df):
    '''
    计算相对湿润指数(近30天就是按每月计算)，
    公式里面的PET使用Thornthwaite方法/高桥蒸发方法计算
    '''
    # # 计算PET
    # df['PET'] = 3100*df['TEM_Avg']/(3100+1.8*(df['PRE_Time_2020']**2)*np.exp((-34.4*df['TEM_Avg'])/(235+df['TEM_Avg'])))
    # # 计算MI
    # df['mi'] = (df['PRE_Time_2020'] - df['PET'])/df['PET']
    # df['mi'] = df['mi'].fillna(0)
    # df['mi'] = np.where(df['mi']<0, 0, df['mi'])

    H = df['TEM_Avg'].resample('A').apply(lambda x: np.sum((x / 5)**1.514)).to_frame()
    H.columns = ['H']
    df = pd.concat([df, H], axis=1)
    df.fillna(method='pad', inplace=True)
    df['A'] = 6.75 * 1e-7 * (df['H']**3) - 7.71 * 1e-5 * (df['H']**2) + 1.7928 * 1e-2 * df['H'] + 0.49
    df['PET'] = 16 * (((10 * df['TEM_Avg']) / df['H'])**df['A'])
    # df['PET'] = df['PET'].fillna(0)

    # 计算MI
    df['mi'] = (df['PRE_Time_2020'] - df['PET']) / df['PET']
    df['mi'] = df['mi'].fillna(0)
    # df['mi'] = np.where(df['mi'] < 0, 0, df['mi'])

    return df


def calc_spi(df, period, weigth_flag=0):
    '''
    TODO 未来改成计算日数据
    
    输入月降水和周期，计算SPI和带权重的SPIW
    flag:0--SPI 1--SPIW
    '''
    if weigth_flag == 0:
        df['cum_pre'] = df['PRE_Time_2020'].rolling(window=period).sum().round(1)
    else:
        assert period == 2, '暂时固定写死period=2的时候，计算SPIW'
        alpha = 0.85  # 规范中计算SPIW,period=2的时候，取值0.85
        df['cum_pre'] = df['PRE_Time_2020'].rolling(window=2).apply(lambda x: (alpha**1) * x[0] + (alpha**2) * x[1])

    # apply_func版本
    def sample(x):
        x = x.to_frame().T
        val = x['cum_pre'].values[0]  # float类型
        month = x.index.month[0]  # int类型

        if np.isnan(val):
            SPI = np.nan
        else:
            data = df.loc[df.index.month == month, 'cum_pre']
            data.dropna(inplace=True)
            a, loc, scale = gamma.fit(data)
            gamma_prob = gamma.cdf(val, a, loc, scale)
            SPI = norm.ppf(gamma_prob)
            SPI = round(SPI, 5)
        return SPI

    df['SPI'] = df.apply(sample, axis=1)
    spi_list = df['SPI'].tolist()

    # 循环版
    # spi_list = []
    # for t in df.index:
    #     val = df.loc[t,'cum_pre'] # 累积降水量

    #     if np.isnan(val):
    #         SPI = np.nan
    #     else:
    #         data = df.loc[df.index.month==t.month, 'cum_pre']
    #         data.dropna(inplace=True)
    #         a, loc, scale = gamma.fit(data)
    #         gamma_prob = gamma.cdf(val, a, loc, scale)
    #         SPI = norm.ppf(gamma_prob)
    #         SPI = round(SPI, 5)

    #     spi_list.append(SPI)

    return spi_list


def calc_mci(df, a, b, c, d):
    '''
    使用日数据计算
    a北方西部0.3 南方0.5
    b北方西北0.5 南方0.6
    c北方西北0.3 南方0.2
    d北方西北0.2 南方0.1
    '''
    df_cp = df.copy()
    
    Ka = [0, 0, 0, 0.6, 1.0, 1.2, 1.2, 1.0, 0.9, 0.4, 0, 0]  # 附录H
    df_cp['Ka'] = np.nan
    for mon in range(1, 13):
        df_cp.loc[df_cp.index.month == mon, 'Ka'] = Ka[mon - 1]

    df_cp = calc_mi(df_cp)
    spiw60 = calc_spi(df_cp, 2, weigth_flag=1)
    spi90 = calc_spi(df_cp, 3)
    spi150 = calc_spi(df_cp, 5)

    df_cp['spiw60'] = spiw60
    df_cp['spi90'] = spi90
    df_cp['spi150'] = spi150
    df_cp['干旱指数'] = df_cp['Ka'] * (a * df_cp['spiw60'] + b * df_cp['mi'] + c * df_cp['spi90'] + d * df_cp['spi150'])
    df_cp['干旱指数'] = df_cp['干旱指数'].fillna(0).round(2)
    df_cp['干旱指数'] = df_cp['干旱指数'].replace(-0.0, 0.0)
    df_cp.dropna(subset=['TEM_Avg'], inplace=True)

    mci = df_cp['干旱指数'].to_frame()
    mci['轻度干旱'] = np.where((mci['干旱指数'] > -1) & (mci['干旱指数'] <= -0.5), 1, 0)
    mci['中度干旱'] = np.where((mci['干旱指数'] > -1.5) & (mci['干旱指数'] <= -1), 1, 0)
    mci['重度干旱'] = np.where((mci['干旱指数'] > -2) & (mci['干旱指数'] <= -1.5), 1, 0)
    mci['特度干旱'] = np.where(mci['干旱指数'] <= -2, 1, 0)

    return mci


# path = r'C:/Users/mjynj/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
# df = pd.read_csv(path, low_memory=False)
# df = df[df['Station_Id_C'] == 52866]
# df = df[['Datetime', 'Station_Id_C', 'Lon', 'Lat', 'TEM_Avg', 'PRE_Time_2020']]
# df['Datetime'] = pd.to_datetime(df['Datetime'])
# df.set_index('Datetime', inplace=True)
# df.fillna(0, inplace=True)
# mci = calc_mci(df, 0.3, 0.5, 0.3, 0.2)

