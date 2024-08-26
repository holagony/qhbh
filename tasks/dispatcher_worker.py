import sys
import requests
import logging
import json
import simplejson
from celery.result import AsyncResult
from flask import Blueprint, jsonify
from tasks.celery_factory import my_celery
from Utils.name_utils import *

bp_tasks = Blueprint('tasks', __name__)
log = logging.getLogger(__name__)

sys.path.append("Module00")
sys.path.append("Module01")
sys.path.append("Module02")
sys.path.append("Module03")
sys.path.append("Module04")
sys.path.append("Module05")
sys.path.append("Module06")
sys.path.append("Module07")
sys.path.append("Module08")
sys.path.append("Module09")
sys.path.append("Module10")
sys.path.append("Module11")
sys.path.append("Module12")
sys.path.append("Module13")
sys.path.append("Utils")
sys.path.append(".")


def callback(url, result_id, result):
    if url is None:
        return
    header = {'Content-Type': 'application/json'}
    _json = {"id": result_id, "status": "fail", "results": result}
    requests.put(url, headers=header, data=json.dumps(_json))


@my_celery.task(name='dispatcher_actor_qhbh')
def celery_submit(actor_class_str, json_str):
    try:
        data_json = json.loads(json_str)
        result_id = data_json.get('id')
        callback_url = data_json.get('callback')
        module_str = name_convert_to_snake(actor_class_str)
        class_str = name_convert_to_camel(actor_class_str)
        actor_class = getattr(__import__(module_str), class_str)()
        task_result = actor_class.act(json_str)
        return task_result

    except Exception as e:
        response = {'code': 500, 'msg': str(e)}
        return_data = simplejson.dumps({'code': 500, 'msg': str(e), 'data': {}}, ensure_ascii=False, ignore_nan=True)
        log.info(f'dispatcher_worker.py中celery抛异常{response}')
        callback(callback_url, result_id, return_data)
        raise


@bp_tasks.route('/status/<task_id>')
def celery_task_status(task_id):
    async_result = celery_submit.AsyncResult(task_id)
    response = {}
    # async_result.status
    # async_result.state
    # None
    # async_result.info
    # None
    # async_result.result
    # 此操作会阻塞
    # async_result.get()

    if async_result.successful():
        result = async_result.get()
        log.info(f'查询出的异步计算结果：{result}')

        if result is None:
            response = {'code': 500, 'msg': '计算失败', 'data': result}
        else:
            response = {'code': 200, 'msg': '计算成功', 'data': result}
        return jsonify(response)
        # result.forget()  # 将结果删除

    elif async_result.failed():
        log.info('执行失败')
        response = {'code': 500, 'msg': f"任务执行失败,{async_result.result}", 'data': {}}
        return jsonify(response)

    elif async_result.state == 'PENDING':
        # job did not start yet
        response = {'code': 202, 'msg': '任务正在执行，Pending...', 'data': {}, 'state': async_result.state, 'current': 0, 'total': 1, 'status': 'Pending...，任务等待中被执行'}
        return jsonify(response)

    elif async_result.status == 'RETRY':
        log.info('任务异常后正在重试')

    elif async_result.status == 'STARTED':
        log.info('任务已经开始被执行')
        response = {'code': 202, 'msg': 'started...，任务已经开始执行', 'data': {}, 'state': async_result.state, 'current': 0, 'total': 1, 'status': 'started...，任务已经开始执行'}

    elif async_result.state != 'FAILURE':
        pass

    else:
        # something went wrong in the background job
        response = {'code': 500, 'msg': "error", 'data': {}, 'state': async_result.state, 'current': 1, 'total': 1, 'status': str(async_result.info)}

    return jsonify(response)


def get_plugin_by_name(self, package, clazz_name):
    """
    package 类所在路径
    clazz_name 类名
    http://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
    https://docs.python.org/2/library/functions.html?highlight=__import__#__import__
    :param package:
    :param clazz_name:
    :return:
    """
    try:
        plugin_name = clazz_name.capitalize() + 'Plugin'
        module = __import__(package + '.' + clazz_name, fromlist=[plugin_name])
        clazz = getattr(module, plugin_name)
        return clazz(region_id=self.get_current_region_id(), access_key=self.aliyun_access_key)

    except Exception as exp:
        log.error("Get plugin: %s by package: %s error for %s" % (clazz_name, package, exp))
