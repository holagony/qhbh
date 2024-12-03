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

if os.name == 'nt':
    flag = 'local'
elif os.name == 'posix':
    flag = 'HX'
else:
    flag = 'HX'
                
# flag = 'local'

# 信息配置
__C.INFO = edict()
__C.INFO.HX_NODE = 'HXC1116.localdomain'
__C.INFO.NUM_THREADS = 20  # 多线程数量

__C.INFO.IN_UPLOAD_FILE = '/zipdata'
__C.INFO.OUT_UPLOAD_FILE = '/mnt/PRESKY/project/bgdb/qihou/zipdata' if flag == 'HX' else 'C:/Users/MJY/Desktop/qhbh/zipdata'

__C.INFO.IN_DATA_DIR = '/data' # if flag == 'HX' else 'C:/Users/MJY/Desktop/qhbh/data' # 容器内保存文件夹
__C.INFO.OUT_DATA_DIR = '/mnt/PRESKY/project/bgdb/qihou/data' if flag == 'HX' else 'C:/Users/MJY/Desktop/qhbh/data'  # 容器外挂载保存文件夹
__C.INFO.OUT_DATA_URL = 'http://221.122.67.145:8889/qh_climate/result/' if flag == 'HX' else 'http://221.122.67.145:8889/qh_climate/result/'

__C.INFO.REDIS_HOST = '192.168.1.119' if flag == 'HX' else '172.17.0.1'
__C.INFO.REDIS_PORT = '8086' if flag == 'HX' else '6379'
__C.INFO.REDIS_PWD = 'hC%34okFq&' if flag == 'HX' else ''

__C.INFO.DB_USER = 'postgres'
__C.INFO.DB_PWD = '2023p+yuiL34gf+hx+##!!'
__C.INFO.DB_HOST = '192.168.1.122' if flag == 'HX' else '1.119.169.101'
__C.INFO.DB_PORT = '5432' if flag == 'HX' else '10089'
__C.INFO.DB_NAME = 'postgres'
__C.INFO.SCHEMA_NAME = 'public'

__C.INFO.READ_LOCAL = True if flag == 'HX' else True
__C.INFO.SAVE_RESULT = False if flag == 'HX' else False
__C.INFO.MAPBOX_TOKEN = 'pk.eyJ1IjoiZGFpbXUiLCJhIjoiY2x3MWV6Y3YxMDF5aDJxcWI2c3c3eWh4dSJ9.DWzNsJKgNetnDZi4ZKV2Yg'
__C.INFO.TILE_PATH = os.path.join(data_file_dir, 'mapbox_tile/') if flag == 'HX' else os.path.join(data_file_dir, 'mapbox_tile\\')


# 样例数据文件路径
__C.FILES = edict()
__C.FILES.FONT = os.path.join(data_file_dir, 'fonts/simhei.ttf')

# 站点信息
__C.FILES.STATION = os.path.join(data_file_dir, 'qh_station.csv')

__C.FILES.IDW_W = os.path.join(data_file_dir, 'idw/idw.dll')
__C.FILES.IDW_L = os.path.join(data_file_dir, 'idw/libidw.so')

# 行业数据
__C.FILES.FILE01 = os.path.join(data_file_dir, '行业数据/01_GDP.xlsx')
__C.FILES.FILE02 = os.path.join(data_file_dir, '行业数据/02_人口.xlsx')
__C.FILES.FILE03 = os.path.join(data_file_dir, '行业数据/03_能源.xlsx')
__C.FILES.FILE04 = os.path.join(data_file_dir, '行业数据/04_交通.xlsx')

# 青海边界文件
__C.FILES.BOUNDARY = os.path.join(data_file_dir, '青海边界')



# shp文件路径
__C.FILES.LAKE = os.path.join(data_file_dir, 'shp/lake.shp')
__C.FILES.ICE = os.path.join(data_file_dir, 'shp/ice.shp')

# 气候变化风险预估-降水/干旱
__C.FILES.DISASTER = os.path.join(data_file_dir, 'disaster.nc') # 承灾体数据
__C.FILES.DROUGHT_CZT = os.path.join(data_file_dir, 'drought_czt.nc') # 承灾体数据
__C.FILES.DROUGHT_YZ = os.path.join(data_file_dir, 'drought_yz.nc') # 承灾体数据
__C.FILES.DROUGHT_GDP = os.path.join(data_file_dir, 'drought_gdp.nc') # 承灾体数据







