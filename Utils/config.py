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

# shp文件路径
__C.FILES.LAKE = os.path.join(data_file_dir, 'shp/lake.shp')
__C.FILES.ICE = os.path.join(data_file_dir, 'shp/ice.shp')

# 气候变化风险预估-降水 承载体数据
__C.FILES.DISASTER = os.path.join(data_file_dir, 'disaster.nc')



# 内涝
__C.INFO.PRODUCT_RESIZE = False # 是否对结果插值到0.001度
__C.INFO.SAVE_PRE = True # 是否额外生成降水网格数据 (将原始降水数据处理成和积水深度结果相同时间尺度和网格大小的nc)


# 静态数据设置
__C.FILES.DEM_TY = os.path.join(data_file_dir, 'data/dem_ty.tif') # 静态dem数据
__C.FILES.LANDUSE_TY = os.path.join(data_file_dir, 'data/landuse_ty.tif') # 静态土地利用数据
__C.FILES.WATERSH_TY = os.path.join(data_file_dir, 'data/watersh_ty.tif') # 静态集水区数据

__C.FILES.DEM_SHANXI = os.path.join(data_file_dir, 'data/dem_shanxi.tif') # 静态dem数据
__C.FILES.LANDUSE_SHANXI = os.path.join(data_file_dir, 'data/landuse_shanxi.tif') # 静态土地利用数据
__C.FILES.WATERSH_SHANXI = os.path.join(data_file_dir, 'data/watersh_shanxi.tif') # 静态集水区数据

__C.FILES.TY_ROAD_LEVEL1 = os.path.join(data_file_dir, 'road_points/ty_level1.geojson') # 太原1级道路
__C.FILES.TY_ROAD_LEVEL2 = os.path.join(data_file_dir, 'road_points/ty_level2.geojson') # 太原2级道路

__C.FILES.SHANXI_ROAD_LEVEL1 = os.path.join(data_file_dir, 'road_points/ty_level1.geojson') # 山西1级道路
__C.FILES.SHANXI_ROAD_LEVEL1 = os.path.join(data_file_dir, 'road_points/ty_level1.geojson') # 山西2级道路

# 参数设置
__C.PARAMS = edict()
__C.PARAMS.PRE_PROCESS_METHOD = 'nearest'  # 降水网格数据统一分辨率插值方法，可选：nearest/linear/cubic
__C.PARAMS.CITY_DSM_OFFSET = 30  # 城市下垫面高度补偿值 单位：m
__C.PARAMS.TOWM_DSM_OFFSET = 10  # 乡村下垫面高度补偿值 单位：m
__C.PARAMS.LANDUSE_ROAD = 98  # 道路CN
__C.PARAMS.LANDUSE_11 = 70  # 水田CN
__C.PARAMS.LANDUSE_12 = 90  # 旱地CN
__C.PARAMS.LANDUSE_21 = 60  # 有林地CN
__C.PARAMS.LANDUSE_22 = 70  # 灌木林地CN
__C.PARAMS.LANDUSE_23 = 75  # 疏林地CN
__C.PARAMS.LANDUSE_24 = 75  # 其他林地CN
__C.PARAMS.LANDUSE_31 = 73  # 高覆盖度草地CN
__C.PARAMS.LANDUSE_32 = 66  # 中覆盖度草地CN
__C.PARAMS.LANDUSE_33 = 54  # 低覆盖度草地CN
__C.PARAMS.LANDUSE_41 = 100  # 河渠CN
__C.PARAMS.LANDUSE_42 = 100  # 湖泊CN
__C.PARAMS.LANDUSE_43 = 85  # 水库、坑塘CN
__C.PARAMS.LANDUSE_44 = 100  # 冰川永久积雪CN
__C.PARAMS.LANDUSE_45 = 81  # 海涂CN
__C.PARAMS.LANDUSE_46 = 75  # 滩地CN
__C.PARAMS.LANDUSE_51 = 95  # 城镇CN
__C.PARAMS.LANDUSE_52 = 88  # 农村居名点CN
__C.PARAMS.LANDUSE_53 = 98  # 工交建设用地CN
__C.PARAMS.LANDUSE_61 = 85  # 沙地CN
__C.PARAMS.LANDUSE_62 = 92  # 戈壁CN
__C.PARAMS.LANDUSE_63 = 83  # 盐碱地CN
__C.PARAMS.LANDUSE_64 = 89  # 沼泽地CN
__C.PARAMS.LANDUSE_65 = 86  # 裸土地CN
__C.PARAMS.LANDUSE_66 = 90  # 裸岩石砾地CN
__C.PARAMS.LANDUSE_67 = 95  # 其他未利用地CN
__C.PARAMS.DRAINAGE = 5.327  # 排水能力因子 单位：mm
__C.PARAMS.CN_80_TO_90_OFFSET = 0.7  # 80~90CN区域的排水能力削弱系数
__C.PARAMS.CN_OVER_90_OFFSET = 0.8  # 90CN以上区域的排水能力削弱系数
__C.PARAMS.SCS_ALPHA_FACTOR = 1.05  # SCS-CN中计算蓄水S值的alpha系数
__C.PARAMS.SCS_BETA_FACTOR = 1.11  # SCS-CN中计算蓄水S值的beta系数
__C.PARAMS.SCS_LAMBDA = 0.05  # SCS-CN中计算径流的lambda系数
__C.PARAMS.DEPTH_SCALE = 3.52  # 系数






