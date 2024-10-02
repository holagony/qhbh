import warnings
warnings.filterwarnings('ignore')

import collections
import numpy as np
from tqdm import tqdm
from typing import Tuple
from Utils.config import cfg

Land_CN = {
    # 土地利用数据里面融合全国1/2/3/4级道路，对应数值5~10
    5: cfg.PARAMS.LANDUSE_ROAD,
    6: cfg.PARAMS.LANDUSE_ROAD,
    7: cfg.PARAMS.LANDUSE_ROAD,
    8: cfg.PARAMS.LANDUSE_ROAD,
    9: cfg.PARAMS.LANDUSE_ROAD,
    10: cfg.PARAMS.LANDUSE_ROAD,
    11: cfg.PARAMS.LANDUSE_11,  # 水田
    12: cfg.PARAMS.LANDUSE_12,  # 旱地
    21: cfg.PARAMS.LANDUSE_21,  # 有林地
    22: cfg.PARAMS.LANDUSE_22,  # 灌木林地
    23: cfg.PARAMS.LANDUSE_23,  # 疏林地
    24: cfg.PARAMS.LANDUSE_24,  # 其他林地
    31: cfg.PARAMS.LANDUSE_31,  # 高覆盖度草地
    32: cfg.PARAMS.LANDUSE_32,  # 中覆盖度草地
    33: cfg.PARAMS.LANDUSE_33,  # 低覆盖度草地
    41: cfg.PARAMS.LANDUSE_41,  # 河渠
    42: cfg.PARAMS.LANDUSE_42,  # 湖泊
    43: cfg.PARAMS.LANDUSE_43,  # 水库、坑塘
    44: cfg.PARAMS.LANDUSE_44,  # 冰川永久积雪
    45: cfg.PARAMS.LANDUSE_45,  # 海涂
    46: cfg.PARAMS.LANDUSE_46,  # 滩地
    51: cfg.PARAMS.LANDUSE_51,  # 城镇
    52: cfg.PARAMS.LANDUSE_52,  # 农村居名点
    53: cfg.PARAMS.LANDUSE_53,  # 工交建设用地
    61: cfg.PARAMS.LANDUSE_61,  # 沙地
    62: cfg.PARAMS.LANDUSE_62,  # 戈壁
    63: cfg.PARAMS.LANDUSE_63,  # 盐碱地
    64: cfg.PARAMS.LANDUSE_64,  # 沼泽地
    65: cfg.PARAMS.LANDUSE_65,  # 裸土地
    66: cfg.PARAMS.LANDUSE_66,  # 裸岩石砾地
    67: cfg.PARAMS.LANDUSE_67  # 其他未利用地
}


class SCS_CN:

    def __init__(self, landuse, pre_freq):
        # assert landuse.shape in [(1479,1818)], '地理数据尺寸错误'
        self.landuse = landuse
        self.freq = pre_freq  # 降水时长：小时
        self.drainage = cfg.PARAMS.DRAINAGE
        self.S = self.get_s()
        self.drainage_array = self.drainage_deal()

    def get_s(self):
        land_use = self.landuse
        cn = land_use.copy() * 0
        for i in list(Land_CN):
            cn[land_use == i] = Land_CN[i]
        s = cfg.PARAMS.SCS_ALPHA_FACTOR * ((25400 / cn - 254)**cfg.PARAMS.SCS_BETA_FACTOR)

        return s

    def drainage_deal(self):
        land_use = self.landuse
        drainage_array = np.ones(land_use.shape) * self.drainage * 0.1
        cn = land_use.copy() * 0
        for i in list(Land_CN):
            cn[land_use == i] = Land_CN[i]

        drainage_array[cn > 80] = self.drainage * cfg.PARAMS.CN_80_TO_90_OFFSET
        drainage_array[cn > 90] = self.drainage * cfg.PARAMS.CN_OVER_90_OFFSET
        drainage_array[cn == 100] = self.drainage * 0.1

        return drainage_array

    def calc_runoff(self, pre, last_water_depth):
        land_use = self.landuse
        cn = land_use.copy() * 0
        for i in list(Land_CN):
            cn[land_use == i] = Land_CN[i]

        land_mask = land_use.copy() * 0 + 1
        last_water_depth[np.isnan(last_water_depth)] = 0
        P = pre
        Q = np.zeros_like(P)
        temp_up = (P - cfg.PARAMS.SCS_LAMBDA * self.S)**2
        temp_down = P + (1 - cfg.PARAMS.SCS_LAMBDA) * self.S
        temp_down[temp_down == 0] = 1
        temp_matrix = temp_up / temp_down
        Q[P >= (cfg.PARAMS.SCS_LAMBDA * self.S)] = temp_matrix[P >= (cfg.PARAMS.SCS_LAMBDA * self.S)]
        # print('max Q :',np.max(temp_up), np.max(Q))

        # land_mask[cn == 100] = 0
        now_runoff = (Q + last_water_depth - self.drainage_array * self.freq) * land_mask
        now_runoff[np.isnan(now_runoff)] = 0
        now_runoff[now_runoff <= 0] = 0

        return now_runoff


class RFSM:

    def __init__(self, dem_data, landuse_data, watersh_data, runoff, elevations, row_loc, clo_loc):
        self.dem_data = dem_data
        self.landuse_data = landuse_data
        self.watersh_data = watersh_data
        self.runoff = runoff
        self.elevations = elevations
        self.row_loc = row_loc
        self.clo_loc = clo_loc
        self.mask = self.get_mask()

    def get_mask(self):
        land_use = self.landuse_data
        cn = np.zeros_like(land_use)
        for i in list(Land_CN):
            cn[land_use == i] = Land_CN[i]
        land_mask = np.zeros(land_use.shape)
        land_mask[cn < 100] = 1
        return land_mask

    def cal_storage(self, elevations):
        # unique, count = np.unique(elevations, return_counts=True) # 统计相同水位值序列
        # temp = dict(zip(unique, count))
        # temp_elev = np.asarray(sorted(temp.keys()))
        # temp_area = np.array([temp[x] for x in temp_elev])

        unique, count = np.unique(elevations, return_counts=True)
        temp = dict(zip(unique, count))
        elev1 = np.asarray(sorted(temp.keys()))

        if np.isnan(elev1[-1]) == 1:
            num = count[-1]
            temp_elev = np.zeros([len(elev1) + num - 1])
            temp_area = np.zeros([len(elev1) + num - 1])
            temp_elev = np.where(temp_elev, temp_elev, np.nan)
            if np.isnan(elevations[0]) == 1 and np.isnan(elevations[-1]) == 1:

                temp_elev[1:len(elev1) + 1] = elev1
                temp_area[1:len(elev1)] = count[0:-1]

            elif np.isnan(elevations[0]) == 1 and np.isnan(elevations[-1]) != 1:
                temp_elev[-1] = elev1[-2]
                temp_elev[1:len(elev1) - 1] = elev1[0:-2]

                temp_area[-1] = count[-2]
                temp_area[1:len(elev1) - 1] = count[0:-2]

            elif np.isnan(elevations[0]) != 1 and np.isnan(elevations[-1]) == 1:
                temp_elev[0:len(elev1)] = elev1
                temp_area[0:len(elev1) - 1] = count[0:-1]

            else:
                temp_elev[0] = elev1[0]
                temp_elev[-1] = elev1[-2]
                temp_elev[1:len(elev1) - 2] = elev1[1:-2]

                temp_area[0] = count[0]
                temp_area[-1] = count[-2]
                temp_area[1:len(elev1) - 2] = count[1:-2]

        else:
            temp_elev = elev1
            temp_area = np.array([temp[x] for x in temp_elev])

        temp_storage = temp_elev * np.add.accumulate(temp_area) - np.add.accumulate(temp_elev * temp_area)

        return np.vstack((temp_elev, temp_area, temp_storage))

    def cal_single_water_depth(self, elevation, runoff):
        elev, area, storage = self.cal_storage(list(elevation))

        if elevation.size == 1:
            single_water_level = runoff + elev
        else:
            if runoff <= storage[0]:
                single_water_level = elev[0]
            elif runoff < storage[-1]:
                single_water_level = np.interp(runoff, storage, elev)
            else:
                single_water_level = (runoff - storage[-1]) / np.sum(area) + elev[-1]

        single_water_depth = single_water_level - elevation
        single_water_depth[np.isnan(single_water_depth)] = 0
        single_water_depth[single_water_depth <= 0] = 0

        return single_water_depth

    def cal_water_depth(self):
        water_depth = self.runoff.copy() * 0
        for i in range(len(self.elevations)):
            runoff_i = np.nansum(self.runoff[self.watersh_data == i + 1])
            if runoff_i == 0:
                water_depth[self.row_loc[i], self.clo_loc[i]] = 0
            else:
                single_water_depth = self.cal_single_water_depth(self.elevations[i], runoff_i)
                water_depth[self.row_loc[i], self.clo_loc[i]] = single_water_depth

        water_depth[np.isnan(water_depth)] = 0
        water_depth[water_depth < 0] = 0
        water_depth *= self.mask

        return water_depth
