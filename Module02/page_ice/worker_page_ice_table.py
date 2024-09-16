# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:21:57 2024

@author: Daimu
"""

import json
import simplejson
import requests
from flask import Blueprint, request, jsonify
from Utils.config import cfg
from Utils.ordered_easydict import OrderedEasyDict as edict
from Module02.page_ice.page_ice_table_handler import ice_table_def


def callback(url, result_id, result):
    header = {'Content-Type': 'application/json'}
    _json = {"id": result_id, "status": "finish", "results": result}
    if url is None:
        return
    requests.put(url, headers=header, data=json.dumps(_json))


class workerPageIceTable:

    def act(self, json_str):
        data_json = json.loads(json_str)
        result_id = data_json.get('id')
        callback_url = data_json.get('callback')
        result_dict = ice_table_def(data_json)
        return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
        callback(callback_url, result_id, return_data)
        return return_data