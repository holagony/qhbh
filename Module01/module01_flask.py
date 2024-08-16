import json
import simplejson
from flask import Blueprint, request, jsonify, current_app
from Module01.pageA_handler import statistical_climate_features
# from Module01.page2_handler import 
# from Module01.page3_handler import 
# from Module01.page4_handler import 
# from Module01.page5_handler import 
# from Module01.page6_handler import 

from tasks.dispatcher_worker import celery_submit, celery_task_status

module01 = Blueprint('module01', __name__)


@module01.route('/v1/statistical_climate_features', methods=['POST'])
def statistical_climate_features_ds():
    '''
    查询统计-气候要素接口
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerStatisticalClimateFeatures', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = statistical_climate_features(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data
