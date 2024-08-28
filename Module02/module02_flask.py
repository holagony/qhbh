import json
import simplejson
from flask import Blueprint, request, jsonify, current_app
from tasks.dispatcher_worker import celery_submit, celery_task_status
from Module02.pageA_handler import energy_winter_heating


module02 = Blueprint('module02', __name__)


@module02.route('/v1/energy_winter_heating', methods=['POST'])
def pagea_stats():
    '''
    查询统计-气候要素接口
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerPageA', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = energy_winter_heating(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data
