import os
import simplejson
import numpy as np
import pandas as pd
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict


def get_url_path(base_path, all_result):
    '''
    param base_path 基础路径
    param all_result
    '''
    data_url_dict = edict()

    if len(all_result) != 0:
        for result in all_result:  # result是一个dict
            for key, value in result.items():
                if value is not None:
                    value = pd.DataFrame(value)
                    save_path = os.path.join(base_path, key + '.csv')
                    value.to_csv(save_path, encoding='utf_8_sig')
                    save_path_url = save_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_URL)
                    data_url_dict[key] = save_path_url
        return data_url_dict

    else:
        return None


def add_url_prefix(dict_data, key_name='img_path', old_prefix=cfg.INFO.OUT_DATA_DIR, new_prefix=cfg.INFO.OUT_DATA_URL):
    for key, value in dict_data.items():
        if key == key_name:
            if isinstance(value, list):
                for i in range(len(value)):
                    dict_data[key][i] = value[i].replace(old_prefix, new_prefix)
            else:
                dict_data[key] = value.replace(old_prefix, new_prefix)
        elif isinstance(value, dict):
            add_url_prefix(value, key_name, old_prefix, new_prefix)


if __name__ == '__main__':
    data = {"level1": {"img_path": "/data/pic1.png", "level2": {"img_path": "/data/pic2.png"}, "level3": {"img_path": ["/data/pic4.png", "/data/pic5.png", "/data/pic6.png"]}}}
    add_url_prefix(data, old_prefix='/data')
    print(data)

    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': data}, ensure_ascii=False, ignore_nan=True)
    print(return_data)
