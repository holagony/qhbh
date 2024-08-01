import json
import simplejson
from flask import Blueprint, request, jsonify, current_app
from Module03.module03_handler import weather_phenomena_days, init_and_end_days, weather_process_stats
from tasks.dispatcher_worker import celery_submit, celery_task_status

module03 = Blueprint('module03', __name__)


@module03.route('/v1/feature_stats', methods=['POST'])
def feature_stats_1():
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    type = data_json['type']
    is_async = data_json.get('is_async')

    if type == 1:
        if is_async == 1 or is_async is True or is_async == '1':
            result = celery_submit.delay('workerHighA', json_str)
            return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})
        else:
            result_dict = weather_phenomena_days(data_json)
            return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)

    elif type == 2:
        if is_async == 1 or is_async is True or is_async == '1':
            result = celery_submit.delay('workerHighB', json_str)
            return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})
        else:
            result_dict = init_and_end_days(data_json)
            return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)

    elif type == 3:
        if is_async == 1 or is_async is True or is_async == '1':
            result = celery_submit.delay('workerHighC', json_str)
            return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})
        else:
            result_dict = weather_process_stats(data_json)
            return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)

    return return_data
