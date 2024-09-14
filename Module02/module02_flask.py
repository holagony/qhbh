import json
import simplejson
from flask import Blueprint, request, jsonify, current_app
from tasks.dispatcher_worker import celery_submit, celery_task_status
from Module02.page_energy.page_energe_heating_handler import energy_winter_heating
from Module02.page_energy.page_energe_wind_handler import energy_wind_power
from Module02.page_energy.page_energe_solar_handler import energy_solar_power
from Module02.page_ice.page_ice_model_handler import ice_model_def
from Module02.page_water.page_water_hbv_handler import hbv_single_calc


module02 = Blueprint('module02', __name__)


@module02.route('/v1/energy_winter_heating', methods=['POST'])
def pagea_stats():
    '''
    重点领域与行业预估-能源影响预估-冬季采暖接口
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerPageEnergeHeating', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = energy_winter_heating(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data

@module02.route('/v1/energy_wind', methods=['POST'])
def pagea_wind():
    '''
    重点领域与行业预估-能源影响预估-风能
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerPageEnergeWind', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = energy_wind_power(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data

@module02.route('/v1/energy_solar', methods=['POST'])
def pagea_solar():
    '''
    重点领域与行业预估-能源影响预估-太阳能
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerPageEnergeSolar', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = energy_solar_power(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data

@module02.route('/v1/ice_model', methods=['POST'])
def pagea_solar():
    '''
    重点领域与行业预估-冰冻圈影响预估-模型构建
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerPageIceModel', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = ice_model_def(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data

@module02.route('/v1/water_source_hbv', methods=['POST'])
def pagea_solar():
    '''
    重点领域与行业预估-水资源影响预估-HBV
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerPageWaterHbv', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = hbv_single_calc(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data