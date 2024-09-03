# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:31:16 2024

@author: EDY
"""

import json
import simplejson
import requests
from flask import Blueprint, request, jsonify
from Utils.config import cfg
from Module02.page_energy.page_energe_wind_handler import energy_wind_power


def callback(url, result_id, result):
    header = {'Content-Type': 'application/json'}
    _json = {"id": result_id, "status": "finish", "results": result}
    if url is None:
        return
    requests.put(url, headers=header, data=json.dumps(_json))


class workerPageEnergeWind:

    def act(self, json_str):
        data_json = json.loads(json_str)
        result_id = data_json.get('id')
        callback_url = data_json.get('callback')
        result_dict = energy_wind_power(data_json)
        return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
        callback(callback_url, result_id, return_data)
        return return_data