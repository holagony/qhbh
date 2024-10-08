import os
import uuid
import numpy as np
import pandas as pd
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Flood.wrapped.calc_flood import flood_model


def flood_calc(data_json):
    '''
    入参组件
    '''
    # 获取参数
    pre_path = data_json.get('pre_path')  # 完整的容器外降水数据路径
    flag = data_json['flag']
    pre_type = data_json['pre_type']
    previous_id = data_json.get('previous_id')
    param_A = data_json.get('param_A')
    param_b = data_json.get('param_b')
    param_C = data_json.get('param_C')
    param_n = data_json.get('param_n')
    r = data_json.get('r')
    p = data_json.get('p')
    t = data_json.get('t')
    total_t = data_json.get('total_t')

    # 参数处理
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)

    if pre_path is not None:
        pre_path = pre_path.replace(cfg.INFO.OUT_FLOOD_DATA, cfg.INFO.IN_FLOOD_DATA)  # inupt_path要转换为容器内的路径
        print('读取数据路径：' + pre_path)

    if previous_id is not None:
        previous_id = os.path.join(cfg.INFO.IN_DATA_DIR, previous_id)

    sf = flood_model(data_dir, flag, pre_path, pre_type, previous_id, param_A, param_b, param_C, param_n, r, p, t, total_t)
    result_dict = sf.calc_flood()
    result_dict['uuid'] = uuid4

    return result_dict
