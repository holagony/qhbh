import json
import simplejson
from flask import Blueprint, request, jsonify, current_app
from Flood.flood_handler import flood_calc
from tasks.dispatcher_worker import celery_submit, celery_task_status

floodCalc = Blueprint('floodCalc', __name__)


@floodCalc.route('/v1/flood', methods=['POST'])
def flood():
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')

    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerFlood', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    else:
        result_dict = flood_calc(data_json)
        return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)

    return return_data
