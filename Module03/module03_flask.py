import os
import uuid
import json
import simplejson
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from Utils.config import cfg
from flask import Blueprint, request, jsonify, current_app
from tasks.dispatcher_worker import celery_submit, celery_task_status
import matplotlib 
from Module03.page_report import page_report

matplotlib.use('agg')

module03 = Blueprint('module03', __name__)


@module03.route('/v1/boundary_png', methods=['POST'])
def boundary_png():
    '''
    获取PNG图片和最大最小经纬度点
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    area_code = data_json['area_code']
    uuid4 = uuid.uuid4().hex
    data_out = os.path.join(cfg.INFO.IN_DATA_DIR, 'Module03', uuid4)
    if not os.path.exists(data_out):
        os.makedirs(data_out)
        os.chmod(data_out, 0o007 | 0o070 | 0o700)

    # 实时流程
    try:
        gdf = gpd.read_file(os.path.join(cfg.FILES.BOUNDARY, area_code+'.shp'))
    except:
        gdf = gpd.read_file(os.path.join(cfg.FILES.BOUNDARY, area_code+'.geojson'))

    bounds = gdf.total_bounds
    fig, ax = plt.subplots()
    gdf.to_crs(crs='EPSG:4326').plot(ax=ax, color='black').set_axis_off()

    min_lon = bounds[0]-0.1
    max_lon = bounds[2]+0.1
    min_lat = bounds[1]-0.1
    max_lat = bounds[3]+0.1

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    png_path = os.path.join(data_out, f'{area_code}.png')
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')

    png_path = png_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 图片容器内转容器外路径
    png_path = png_path.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
    
    info = dict()
    info['area_code'] = area_code
    info['min_lon'] = min_lon
    info['min_lat'] = min_lat
    info['max_lon'] = max_lon
    info['max_lat'] = max_lat
    info['png_path'] = png_path
    
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': info}, ensure_ascii=False, ignore_nan=True)
    return return_data

@module03.route('/v1/page_report', methods=['POST'])
def pagea_report():
    '''
    报告制作
    '''
    json_str = request.get_data(as_text=True)  # 获取JSON字符串
    data_json = json.loads(json_str)
    is_async = data_json.get('is_async')
    if is_async == 1 or is_async is True or is_async == '1':
        result = celery_submit.delay('workerPageReport', json_str)
        return jsonify({'code': 202, 'msg': '任务提交成功，开始计算...', 'data': {'task_id': result.id}})

    result_dict = page_report(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    return return_data