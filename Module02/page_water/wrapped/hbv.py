import numpy as np
import datetime


def module_snowmelting(t, precipitations, temp, snow_pack, rainfall, F=1, temp_threshold=0):
    """
    Snow melting module
    
    Returns the amount of rainfall and snow accumulation.
    
    ------------------------------------------------------------
    
    Parameters:
    
    t, date - Running day
    precipitations, array - Expected precipitations for the running period
    temp, array - Expected temperatures for the running period
    snow_pack, array - Amount of snow accumulated
    F, float - Degree-day factor
    temp_threshold, float - Temperature threshold for snow to melt
    
    输出：该时刻进入到土壤里面的雨雪量,和该时刻积雪
    """

    if temp[t] > temp_threshold:
        # snow melts and precipitations are rainfalls
        # F * (temp[t] - temp_threshold) 融雪速率，乘上积雪量，作为一部分的水量
        rainfall[t] = precipitations[t] + F * (temp[t] - temp_threshold) * snow_pack[t - 1]
        snow_pack[t] = snow_pack[t - 1] * (1 - F * (temp[t] - temp_threshold))
    else:
        # there is no rainfall, precipitations contribute to the snow pack
        snow_pack[t] = snow_pack[t - 1] + precipitations[t]
        rainfall[t] = 0

    return rainfall, snow_pack


def module_soilmoisture(t, rainfall, sm, effective_precipitation, FC=1, beta=1):
    """
    Soil moisture module
    
    Returns the updated soil moisture and the effective precipitation that contributes to the runoff.
        
    ------------------------------------------------------------
    
    Parameters:
    
    t, date - Running day
    rainfall, array - Amount of rain and melting snow that goes to the soil
    sm, array - Soil Moisture water amount
    FC, float - Field Capacity, ie maximum storage in the subsurface zone
    beta, positive integer - Shape Coefficient for the contribution to effective runoff
    
    输出：该时刻的土壤湿度和有效降水
    """

    effective_precipitation[t] = rainfall[t] * (sm[t - 1] / FC)**beta  # 有效降水 rainfall = P + Sm，即上个模块的输出
    sm[t] = sm[t - 1] + rainfall[t] * (1 - (sm[t - 1] / FC)**beta)

    return sm, effective_precipitation


def module_evapotranspiration(t, m, sm, ea, pea, temp, temp_m, pe_m, C=1, PWP=0):
    """
    Evapotranspiration module
    
    Returns the evapotranspiration and the soil moisture
    
    ------------------------------------------------------------
    
    Parameters:
    
    t, date, - Running day
    m, integer between 1 and 12 - Running month
    sm, array - Soil Moisture water amount
    temp, array - Expected temperatures for the running period
    temp_m, array - Long-term monthly mean temperatures
    pe_m, array -  Long-term mean monthly potential evapotranspiration 
    C, float - Model parameter
    PWP, float - Soil Permanent Wilting Poin

    输出：
    ea 真实潜散发
    sm  = sm - ea (算出来的ea会更新sm)
    """

    # compute the daily adjusted potential evapotranspiration
    pea[t] = (1 + C * (temp[t] - temp_m[m])) * pe_m[m]  # 调整后的潜在蒸散量

    # computes actual evapotranspiration
    if sm[t] < PWP:
        ea[t] = pea[t] * sm[t] / PWP
    else:
        ea[t] = pea[t]

    sm[t] -= ea[t]

    return ea, sm


def module_runoff(t, effective_precipitation, rl, ru, a0, a1, a2, K=[1, 1, 1, 1], L=200):
    """
    Runoff module
    
    Returns the three runoff discharges.
    
    ------------------------------------------------------------
    
    Parameters:
    
    t, date, - Running day
    L, float - Upper reservoir threshold 
    K, array - Recessions coefficients for each reservoir
    ru, array - Upper reservoir level
    rl, array - Lower reservoir level
    a0, array - Near surface runoff 
    a1, array - Inflow runoff 
    a2, array - Baseflow runoff 
    perc, float - Quantity that is percolated from the upper to the lower reservoir
    effective_precipitation, array - Effective contribution to the runoff
    
    """
    # inflow in reservoirs 近地表&地下互流
    # s1(i,s)=s1(i-1,s)+dq(i,s)-(max(0,s1(i-1,s)-l(s))*k0(s))-(s1(i-1,s)*k1(s))-(s1(i-1,s)*kp(s));
    # s2(i,s)=s2(i-1,s)+(s1(i-1,s)*kp(s))-s2(i-1,s)*k2(s);
    # q(i,s)=(max(0,s1(i-1,s)-l(s)))*k0(s)+(s1(i,s)*k1(s))+(s2(i,s)*k2(s));
    # qm(i,s)=(q(i,s)*ca*1000)/(24*3600);

    perc = ru[t - 1] * K[3]
    ru[t] = max(0, ru[t - 1] + effective_precipitation[t] - perc)  # 水库1水位更新
    rl[t] = rl[t - 1] + min(ru[t - 1] + effective_precipitation[t], perc)  # 水库2水位更新

    # outflows
    a0[t] = K[0] * max(0, ru[t] - L)
    a1[t] = K[1] * (ru[t] - a0[t]) # 减去a0[t]待观察
    a2[t] = K[2] * rl[t]

    # update reservoirs lvls
    ru[t] = ru[t] - (a0[t] + a1[t])
    rl[t] = rl[t] - a2[t]

    return rl, ru, a0, a1, a2


def total_runoff(t, a0, a1, a2, area):
    """
    Returns the total runoff discharge for one day.
    
    ------------------------------------------------------------
        
    Parameters:
    
    t, date, - Running day
    a0, array - Near surface runoff 
    a1, array - Inflow runoff 
    a2, array - Baseflow runoff 
    
    """
    area = area * 1e6  # 单位：km^2 --> m^2
    q = (a0[t] + a1[t] + a2[t]) * 1e-3  # 单位：mm/d --> m/d
    qm = (q * area) / (24 * 3600)  # 单位：m/d --> m^3/s
    return q, qm


def HBV(days, precipitations, temp, pe_m, temp_m, init, temp_threshold, F, FC, beta, PWP, C, K0, K1, K2, K_prec, L, area):
    """
    Returns the total runoff waters for each day.
        
    ------------------------------------------------------------
        
    Parameters:
    
    days, array of dates - Days of the running period
    precipitations, array - Expected precipitations for the running period
    temp, array - Expected temperatures for the running period
    temp_m, array - Long-term monthly mean temperatures
    pe_m, array -  Long-term mean monthly potential evapotranspiration
    sm0, float - Initial soil moisutre
    init, list - List of the initial values in the following order: sm0, rl0, ru0, snow_pack0 
    
    """

    # Parameters:
    # F = 3  # module1
    # temp_threshold = 0.0  # module1
    # FC = 150  # module2
    # beta = 1  # module2
    # PWP = 150  # module3
    # C = 0.05  # module3 控制蒸散发模块
    # K = [0.2, 0.1, 0.05, 0.05]  # module4 k0初始值应该始终大于k1，k2小于k1, k3是渗流系数
    # L = 5  # module4 水位阈值

    T = len(days)
    K = [K0, K1, K2, K_prec]

    # initialization
    # 初始积雪量 土壤湿度 上游水库水位 下游水库水位
    snow_pack = np.zeros(T)
    sm = np.zeros(T)
    rl = np.zeros(T)
    ru = np.zeros(T)

    rainfall = np.zeros(T)
    effective_precipitation = np.zeros(T)
    total = np.zeros(T)
    total_m = np.zeros(T)

    ea = np.zeros(T)
    pea = np.zeros(T)
    a0 = np.zeros(T)  # 近地表流量
    a1 = np.zeros(T)  # 互流
    a2 = np.zeros(T)  # 基流

    snow_pack0, sm0, rl0, ru0 = init
    rl0 = rl0 * 1e3  # 米转毫米
    ru0 = ru0 * 1e3  # 米转毫米
    L = L * 1e3  # 米转毫米

    snow_pack[0] = snow_pack0
    sm[0] = sm0
    rl[0] = rl0
    ru[0] = ru0

    # loop over days
    for t in range(1, len(days) - 1):
        m = days[t].month
        rainfall, snow_pack = module_snowmelting(t, precipitations, temp, snow_pack, rainfall, F, temp_threshold)
        sm, effective_precipitation = module_soilmoisture(t, rainfall, sm, effective_precipitation, FC, beta)
        ea, sm = module_evapotranspiration(t, m, sm, temp, ea, pea, temp_m, pe_m, C, PWP)
        rl, ru, a0, a1, a2 = module_runoff(t, effective_precipitation, rl, ru, a0, a1, a2, K, L)
        q, qm = total_runoff(t, a0, a1, a2, area)
        total[t] = q
        total_m[t] = qm

    return total, total_m


def nse(predictions, targets):
    nse = 1 - (np.sum((predictions - targets)**2) / np.sum((targets - np.mean(targets))**2))
    return nse


# test
days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(10)]
temp = [5 + i for i in range(len(days) + 1)]  # 气温 单位：度
precipitations = [5 for i in range(len(days) + 1)]  # 单位：mm
pe_m = [10 for i in range(12)]  # 月潜在蒸散发 单位：mm
temp_m = [10 for i in range(12)]  # 月平均气温 这两个参数需要赋值到每一天
snow_pack0, sm0, rl0, ru0 = 0, 0, 0, 0  # rl0, ru0 单位：m
init = [snow_pack0, sm0, rl0, ru0]

temp_threshold = 0.0  # module1 单位：度
F = 3  # module1
FC = 150  # module2 单位：mm
beta = 1  # module2
PWP = 150  # module3 单位：mm
C = 0.05  # module3 控制蒸散发模块
K0 = 0.2
K1 = 0.1
K2 = 0.05
K_prec = 0.05
L = 5  # module4 水位阈值 单位：m
area = 200  # km^2

a_total = HBV(days, precipitations, temp, pe_m, temp_m, init, temp_threshold, F, FC, beta, PWP, C, K0, K1, K2, K_prec, L, area)

# new_month_data = month_data..resample('1D').ffill()
