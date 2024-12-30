import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm


def hbv_main(n_days, date_time, month, air_temp, prec, evp_monthly, tem_monthly, d, fc, beta, c, k0, k1, k2, kp, l, pwp, Tsnow_thresh, ca):

    # Initialize arrays for the simiulation
    snow = np.zeros(air_temp.size)  #
    liq_water = np.zeros(air_temp.size)  #
    pe = np.zeros(air_temp.size)  #
    soil = np.zeros(air_temp.size)  #
    ea = np.zeros(air_temp.size)  #
    dq = np.zeros(air_temp.size)  #
    s1 = np.zeros(air_temp.size)  #
    s2 = np.zeros(air_temp.size)  #
    q = np.zeros(air_temp.size)  #
    qm = np.zeros(air_temp.size)  #

    # Set parameters
    # d = params[0]  #
    # fc = params[1]  #
    # beta = params[2]  #
    # c = params[3]  #
    # k0 = params[4]  #
    # l = params[5]  #
    # k1 = params[6]  #
    # k2 = params[7]  #
    # kp = params[8]  #
    # pwp = params[9]  #
    # Tsnow_thresh = 0.0
    # ca = 150000  # 平方米

    for i_day in range(1, n_days):

        #print i_day
        if air_temp[i_day] < Tsnow_thresh:

            #Precip adds to the snow pack
            snow[i_day] = snow[i_day - 1] + prec[i_day]

            #Too cold, no liquid water
            liq_water[i_day] = 0.0

            #Adjust potential ET base on difference between mean daily temp
            #and long-term mean monthly temp
            year_mon = date_time[i_day].strftime('%Y-%m')
            pe[i_day] = (1. + c * (air_temp[i_day] - tem_monthly[year_mon])) * evp_monthly[year_mon]

            #Check soil moisture and calculate actual evapotranspiration
            if soil[i_day - 1] > pwp:
                ea[i_day] = pe[i_day]
            else:
                #Reduced ET_actual by fraction of permanent wilting point
                ea[i_day] = pe[i_day] * (soil[i_day - 1] / pwp)

            #See comments below
            dq[i_day] = liq_water[i_day] * (soil[i_day - 1] / fc)**beta
            soil[i_day] = soil[i_day - 1] + liq_water[i_day] - dq[i_day] - ea[i_day]
            s1[i_day] = s1[i_day - 1] + dq[i_day] - max(0, s1[i_day - 1] - l) * k0 - (s1[i_day] * k1) - (s1[i_day - 1] * kp)
            s2[i_day] = s2[i_day - 1] + s1[i_day - 1] * kp - s2[i_day] * k2
            q[i_day] = max(0, s1[i_day] - l) * k0 + (s1[i_day] * k1) + (s2[i_day] * k2)
            qm[i_day] = (q[i_day] * ca * 1000.) / (24. * 3600.)
        else:
            #Air temp over threshold: precip falls as rain

            snow[i_day] = max(snow[i_day - 1] - d * air_temp[i_day] - Tsnow_thresh, 0.)

            liq_water[i_day] = prec[i_day] + min(snow[i_day], d * air_temp[i_day] - Tsnow_thresh, 0.)

            #PET adjustment
            year_mon = date_time[i_day].strftime('%Y-%m')
            pe[i_day] = (1. + c * (air_temp[i_day] - tem_monthly[year_mon])) * evp_monthly[year_mon]

            if soil[i_day - 1] > pwp:
                ea[i_day] = pe[i_day]
            else:
                ea[i_day] = pe[i_day] * soil[i_day] / pwp

            #Effective precip (portion that contributes to runoff)
            dq[i_day] = liq_water[i_day] * ((soil[i_day - 1] / fc))**beta

            #Soil moisture = previous days SM + liquid water - Direct Runoff - Actual ET
            soil[i_day] = soil[i_day - 1] + liq_water[i_day] - dq[i_day] - ea[i_day]

            #Upper reservoir water levels
            s1[i_day] = s1[i_day - 1] + dq[i_day] - max(0, s1[i_day - 1] - l) * k0 - (s1[i_day] * k1) - (s1[i_day - 1] * kp)
            #Lower reservoir water levels
            s2[i_day] = s2[i_day - 1] + dq[i_day - 1] * kp - s2[i_day - 1] * k2

            #Run-off is total from upper (fast/slow) and lower reservoirs
            q[i_day] = max(0, s1[i_day] - l) * k0 + s1[i_day] * k1 + (s2[i_day] * k2)
            #Resulting Q
            qm[i_day] = (q[i_day] * ca * 1000.) / (24. * 3600.)

    #End of simulation
    return qm


if __name__ == '__main__':
    pass
    # from Module02.page_water.hbv_handler import hbv_single_calc
    # from Module02.page_water.wrapped.func01_q_stats import stats_q

    # data_json = dict()
    # data_json['time_freq'] = 'Y'
    # data_json['evaluate_times'] = '1950,1980'  # 预估时段时间条
    # data_json['refer_years'] = '2023,2024'  # 参考时段时间条
    # data_json['valid_times'] = '202303,202403'  # 验证期 '%Y%m,%Y%m'
    # data_json['hydro_ids'] = '40100350'  # 唐乃亥
    # data_json['sta_ids'] = '52943,52955,52957,52968,56033,56043,56045,56046,56065,56067'
    # data_json['cmip_type'] = 'original'  # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
    # data_json['cmip_res'] = None  # 分辨率 1/5/10/25/50/100 km
    # data_json['cmip_model'] = ['BCC-CSM2-MR', 'CanESM5']  # 模式，列表：['CanESM5','CESM2']等
    # data_json['degree'] = None
    # data_df, refer_df, data_df_meteo, vaild_cmip, evaluate_cmip = hbv_single_calc(data_json)
    # result = stats_q(data_df, refer_df)

    # # 气象数据处理
    # data_df_meteo['EVP_Taka'] = data_df_meteo['EVP_Taka'].apply(lambda x: 0 if x < 0 else x)
    # data_df_meteo['PRE_Time_2020'] = data_df_meteo['PRE_Time_2020'].fillna(0)
    # data_df_meteo['EVP_Taka'] = data_df_meteo['EVP_Taka'].fillna(0)
    # data_df_meteo.dropna(how='any', axis=0, inplace=True)

    # tem_daily = data_df_meteo.pivot_table(index=data_df_meteo.index, columns=['Station_Id_C'], values='TEM_Avg')  # 统计时段df
    # tem_daily = tem_daily.iloc[:].mean(axis=1).round(2)  # 区域平均，代表流域的平均情况
    # tem_monthly = tem_daily.resample('1M').mean()

    # pre_daily = data_df_meteo.pivot_table(index=data_df_meteo.index, columns=['Station_Id_C'], values='PRE_Time_2020')  # 统计时段df
    # pre_daily = pre_daily.iloc[:].mean(axis=1).round(2)

    # evp_daily = data_df_meteo.pivot_table(index=data_df_meteo.index, columns=['Station_Id_C'], values='EVP_Taka')  # 统计时段df
    # evp_daily = evp_daily.iloc[:].mean(axis=1)
    # evp_monthly = evp_daily.resample('1M').mean().round(2)

    # # input
    # month = tem_daily.index.month.values
    # temp = tem_daily.values  # 气温 单位：度
    # precip = pre_daily.values  # 单位：mm
    # para_init = np.array([6.1, 195, 2.6143, 0.07, 0.163, 4.87, 0.029, 0.049, 0.05, 106])

    # q_sim = hbv_main(len(temp), month, para_init, temp, precip, evp_monthly, tem_monthly)
    # q_sim = pd.DataFrame(q_sim)
    # q_sim['Datetime'] = tem_daily.index
