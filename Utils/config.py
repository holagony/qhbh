import os
from Utils.ordered_easydict import OrderedEasyDict as edict

# 基础路径
os.environ['PROJ_LIB'] = '/home/user/miniconda3/envs/myconda/share/proj'
basedir = os.path.abspath(os.path.dirname(__file__))
current_file = os.path.abspath(__file__)  # 获取当前文件的绝对路径
current_dir = os.path.dirname(current_file)  # 获取当前文件所在目录
current_obj = os.path.dirname(current_dir)  # 获取当前文件所在项目
data_file_dir = os.path.join(current_obj, 'Files')

# 生成字典
__C = edict()
cfg = __C
flag = 'HX'

# 信息配置
__C.INFO = edict()
__C.INFO.HX_NODE = 'HXC1116.localdomain'
__C.INFO.NUM_THREADS = 20  # 多线程数量
__C.INFO.IN_DATA_DIR = '/data'  # 容器内保存文件夹
__C.INFO.OUT_DATA_DIR = '/home/bgdb/docker-test/qhkxxlz/data/' if flag == 'HX' else 'c:/Users/MJY/Desktop/qhkxxlz/data'  # 容器外挂载保存文件夹
__C.INFO.OUT_DATA_URL = 'http://1.119.169.101:10036/img' if flag == 'HX' else 'http://1.119.169.101:10036/img'
__C.INFO.REDIS_HOST = '172.17.0.2' if flag == 'HX' else '172.17.0.2'
__C.INFO.REDIS_PORT = '6379' if flag == 'HX' else '6379'
__C.INFO.REDIS_PWD = '' if flag == 'HX' else 'qhkxxlz123..!@'
__C.INFO.DB_USER = 'postgres'
__C.INFO.DB_PWD = '2023p+yuiL34gf+hx+##!!'
__C.INFO.DB_HOST = '1.119.169.101'
__C.INFO.DB_PORT = '10089'
__C.INFO.DB_NAME = 'postgres'
__C.INFO.SCHEMA_NAME = 'public'
__C.INFO.READ_LOCAL = True if flag == 'HX' else False
__C.INFO.SAVE_RESULT = False if flag == 'HX' else False
__C.INFO.MAPBOX_TOKEN = 'pk.eyJ1IjoiZGFpbXUiLCJhIjoiY2x3MWV6Y3YxMDF5aDJxcWI2c3c3eWh4dSJ9.DWzNsJKgNetnDZi4ZKV2Yg'
__C.INFO.TILE_PATH = os.path.join(data_file_dir, 'mapbox_tile/') if flag == 'HX' else os.path.join(data_file_dir, 'mapbox_tile\\')

# 样例数据文件路径
__C.FILES = edict()
__C.FILES.FONT = os.path.join(data_file_dir, 'fonts/simhei.ttf')
__C.FILES.T_DISTR_TABLE = os.path.join(data_file_dir, 't_distribution_table.csv')
__C.FILES.QH_DATA_HOUR = os.path.join(data_file_dir, 'test_data/qh_hour.csv')
__C.FILES.QH_DATA_DAY = os.path.join(data_file_dir, 'test_data/qh_day.csv')
__C.FILES.QH_DATA_MONTH = os.path.join(data_file_dir, 'test_data/qh_mon.csv')
__C.FILES.QH_DATA_YEAR = os.path.join(data_file_dir, 'test_data/qh_year.csv')
__C.FILES.WIND_TOWER = os.path.join(data_file_dir, 'test_data/格尔木风冷')
__C.FILES.PRE_MIN = os.path.join(data_file_dir, 'test_data/rain_min')
__C.FILES.ADTD = os.path.join(data_file_dir, 'test_data/ADTD2016.csv')
__C.FILES.ADTD_PARAM = os.path.join(data_file_dir, 'test_data/Module10_data/parameters.csv')
__C.FILES.ADTD_FACTOR = os.path.join(data_file_dir, 'test_data/Module10_data/factors.npy')

# report
__C['report']=dict()
__C['report']['picture']=os.path.join(current_obj, 'Report','picture')
__C['report']['template']=os.path.join(current_obj, 'Report','template')
__C['report']['report']=os.path.join(current_obj, 'Report','report')
