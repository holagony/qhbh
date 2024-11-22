import os
import uuid
import time
import xarray as xr
from Utils.config import cfg
from Module01.wrapped.func02_interp_grid import contour_picture
from Module01.wrapped.func03_mk_tests import time_analysis
from Module01.wrapped.func04_cumsum_anomaly import calc_anomaly_cum
from Module01.wrapped.func05_moving_avg import calc_moving_avg
from Module01.wrapped.func06_wavelet_analyse import wavelet_main
from Module01.wrapped.func07_correlation_analysis import correlation_analysis
from Module01.wrapped.func08_eof import eof, reof
from Module01.wrapped.func09_eemd import eemd
from Module01.wrapped.func16_crop_table_stats import agriculture_features_stats

# 农牧业 因为数据不确定，暂时不做

def agriculture_features(data_json):
    '''
    获取天擎数据，参数说明
    :param element：对应原型，传入的要素名称
        作物种植面积：crop_acreage
        产量：yield
        播种期：11 sowin_date
        成熟期：91 maturity:
        生育期：reproductive_period
        生育期天数：reproductive_day
        
    :param crop: 对应原型，作物
        春玉米中熟：010402 spring_maizet
        春小麦：010304 spring_wheat
        冬小麦：010301 winter_wheat
        青稞：010307 highland_barley
        油菜甘蓝型：010603 napus
        油菜白菜型：010602 campestris
        蚕豆：010907 Horsebean
        马铃薯：010906 potato
        农作物：1 crop
        粮食作物：2 food_crop
    :param time_freq: 对应原型选择数据的时间尺度
        传参：
        年 - 'Y'
        有效时间： 2010 - 2024
    :param stats_times: 对应原型的统计时段
        (1)当time_freq选择年Y。下载连续的年数据，传参：'%Y,%Y'

    :param sta_ids: 传入的站点，多站，传：'52866,52713,52714'

    :param interp_method: 对应原型的插值方法
        传参：
        克里金 - 'kriging'
        泛克里金 - 'uni_kriging'
        反距离权重 - 'idw'

    :param ci: 置信区间    
    :param shp_path: shp文件
    :param output_filepath: 输出结果文件

    '''
    # 1.参数读取
    element = data_json['element']
    interp_method = data_json['interp_method']
    shp_path = data_json.get('shp_path')
    sta_ids = data_json['sta_ids']

    # 2.参数处理
    shp_path = shp_path.replace(cfg.INFO.OUT_UPLOAD_FILE, cfg.INFO.IN_UPLOAD_FILE)  # inupt_path要转换为容器内的路径

    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(cfg.INFO.IN_DATA_DIR, uuid4)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)


    result_dict = dict()


    if element in ['sowin_date','maturity']:
        station_df,stats_result,post_data_df,post_refer_df,reg_params,data_r_df=agriculture_features_stats(data_json)

    elif element in ['reproductive_period']:
        stats_result=agriculture_features_stats(data_json)
        stats_result.reset_index(inplace=True)
        stats_result.rename(columns={'Datetime': '年'}, inplace=True)

        result_dict['表格'] = stats_result.to_dict(orient='records')
        
    elif element in ['reproductive_day','crop_acreage','yield']:
        station_df,stats_result,post_data_df,post_refer_df,reg_params=agriculture_features_stats(data_json)

    if element in ['sowin_date','maturity','reproductive_day','crop_acreage','yield']:
        
        if sta_ids!='63000':
            result_dict['uuid'] = uuid4
            result_dict['表格'] = dict()
            result_dict['分布图'] = dict()
            result_dict['统计分析'] = dict()
            
            station_df.columns=['站号','站名','经度','纬度']
            result_dict['站号'] = station_df.to_dict(orient='records')
            result_dict['表格'] = stats_result.to_dict(orient='records')
            result_dict['统计分析']['线性回归'] = reg_params.to_dict(orient='records')
            print('统计表完成')
            if element in ['sowin_date','maturity']:
                result_dict['日期表格'] = data_r_df.to_dict(orient='records')
            data_df=station_df.copy()
            data_df.columns=['Station_Id_C','Station_Name','Lon','Lat']
        
                
            try:
                # 分布图
                nc_path, _, _, _, _ = contour_picture(stats_result, data_df, shp_path, interp_method, data_dir)
                nc_path_trans = nc_path.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)  # 容器内转容器外路径
                nc_path_trans = nc_path_trans.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)  # 容器外路径转url
                result_dict['分布图'] = nc_path_trans
                print('分布图插值生成nc完成')
        
                # 6. 统计分析-EOF分析
                ds = xr.open_dataset(nc_path)
                eof_path = eof(ds, shp_path, data_dir)
                result_dict['统计分析']['EOF分析'] = eof_path
                print('eof完成')
        
                # 7. 统计分析-REOF分析
                ds = xr.open_dataset(nc_path)
                reof_path = reof(ds, shp_path, data_dir)
                result_dict['统计分析']['REOF分析'] = reof_path
                print('reof完成')
            except:
                result_dict['分布图'] = None
                result_dict['统计分析']['EOF分析'] = None
                result_dict['统计分析']['REOF分析'] = None
        
            # 1.统计分析-mk检验
            mk_result = time_analysis(post_data_df, data_dir)
            print('MK检验完成')
        
            # 2.统计分析-累积距平
            anomaly_result = calc_anomaly_cum(post_data_df, post_refer_df, data_dir)
            print('距平完成')
        
            # 3.统计分析-滑动平均
            moving_result = calc_moving_avg(post_data_df, 5, data_dir)
            print('滑动平均完成')
        
            # 4. 统计分析-小波分析
            wave_result = wavelet_main(post_data_df, data_dir)
            print('小波完成')
        
            # 5. 统计分析-相关分析
            correlation_result = correlation_analysis(post_data_df, data_dir)
            print('相关分析完成')
        
            # 8.EEMD分析
            eemd_result = eemd(post_data_df, data_dir)
            print('eemd完成')
        
            # 数据保存
            result_dict['统计分析']['线性回归'] = reg_params.to_dict(orient='records')
            result_dict['统计分析']['MK检验'] = mk_result
            result_dict['统计分析']['累积距平'] = anomaly_result
            result_dict['统计分析']['滑动平均'] = moving_result
            result_dict['统计分析']['小波分析'] = wave_result
            result_dict['统计分析']['相关分析'] = correlation_result
            result_dict['统计分析']['EEMD分析'] = eemd_result
            
        else:
            result_dict['uuid'] = uuid4
            result_dict['表格'] = dict()
            result_dict['表格'] = stats_result.to_dict(orient='records')

    return result_dict


if __name__ == '__main__':
    t1 = time.time()
    data_json = dict()
    data_json['element'] = 'yield' #sowin_date
    data_json['crop'] = 'food_crop'
    data_json['refer_years'] = '2002,2024'
    data_json['nearly_years'] = '2004,2024'
    data_json['time_freq'] = 'Y'
    data_json['stats_times'] = '2008,2024'
    data_json['sta_ids'] = '63000'
    data_json['interp_method'] = 'ukri'
    data_json['ci'] = 95
    data_json['shp_path'] =r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\省界.shp'

    result = agriculture_features(data_json)
    t2 = time.time()
    print(t2 - t1)
