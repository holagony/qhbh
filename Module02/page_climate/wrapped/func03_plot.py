import warnings
warnings.filterwarnings('ignore')

import gc
import math
import numpy as np
import shapely.geometry as sgeom
from shapely.prepared import prep
# import cmaps
import cartopy.crs as ccrs
import matplotlib 
matplotlib.use('agg')

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
from shapely.prepared import prep
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeat
from tqdm import tqdm
from cartopy.io.shapereader import Reader
from Utils.station_to_grid import station_to_grid

    
def polygon_to_mask(polygon, x, y):
    '''
    生成落入多边形的点的掩膜数组，超快版本
    '''
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if x.shape != y.shape:
        raise ValueError('x和y的形状不匹配')
    prepared = prep(polygon)

    def recursion(x, y):
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xflag = math.isclose(xmin, xmax)
        yflag = math.isclose(ymin, ymax)
        mask = np.zeros(x.shape, dtype=bool)

        # 散点重合为单点的情况.
        if xflag and yflag:
            point = sgeom.Point(xmin, ymin)
            if prepared.contains(point):
                mask[:] = True
            else:
                mask[:] = False
            return mask

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        # 散点落在水平和垂直直线上的情况.
        if xflag or yflag:
            line = sgeom.LineString([(xmin, ymin), (xmax, ymax)])
            if prepared.contains(line):
                mask[:] = True
            elif prepared.intersects(line):
                if xflag:
                    m1 = (y >= ymin) & (y <= ymid)
                    m2 = (y >= ymid) & (y <= ymax)
                if yflag:
                    m1 = (x >= xmin) & (x <= xmid)
                    m2 = (x >= xmid) & (x <= xmax)
                if m1.any(): mask[m1] = recursion(x[m1], y[m1])
                if m2.any(): mask[m2] = recursion(x[m2], y[m2])
            else:
                mask[:] = False
            return mask

        # 散点可以张成矩形的情况.
        box = sgeom.box(xmin, ymin, xmax, ymax)
        if prepared.contains(box):
            mask[:] = True
        elif prepared.intersects(box):
            m1 = (x >= xmid) & (x <= xmax) & (y >= ymid) & (y <= ymax)
            m2 = (x >= xmin) & (x <= xmid) & (y >= ymid) & (y <= ymax)
            m3 = (x >= xmin) & (x <= xmid) & (y >= ymin) & (y <= ymid)
            m4 = (x >= xmid) & (x <= xmax) & (y >= ymin) & (y <= ymid)
            if m1.any(): mask[m1] = recursion(x[m1], y[m1])
            if m2.any(): mask[m2] = recursion(x[m2], y[m2])
            if m3.any(): mask[m3] = recursion(x[m3], y[m3])
            if m4.any(): mask[m4] = recursion(x[m4], y[m4])
        else:
            mask[:] = False

        return mask

    return recursion(x, y)


def interp_and_mask(shp_path, lon_list, lat_list, value_list, method):
    '''
    先插值后掩膜，用于后续画图
    shp_path: shp文件路径，也可以是geojson
    lon_list: 站点的经度列表
    lat_list: 站点的纬度列表
    value_list: 从统计的表格中提取，站点的要素值列表
    method: 插值方法
    
    返回:
        mask_grid 掩膜后的结果网格
        lon_grid 掩膜后的经度网格
        lat_grid 掩膜后的纬度网格
    '''
    shp = gpd.read_file(shp_path,encoding='utf-8')
    bounds = shp['geometry'].total_bounds
    lon_max = bounds[2]
    lon_min = bounds[0]
    lat_max = bounds[3]
    lat_min = bounds[1]
    gridx = np.arange(lon_min, lon_max + 0.01, 0.01)
    gridy = np.arange(lat_min, lat_max + 0.01, 0.01)

    # 散点数据插值
    grid = station_to_grid(lon_list, lat_list, value_list, gridx, gridy, method, name=None)

    # 对插值的grid掩膜
    multi_polygon = shp['geometry'].unary_union
    lon_grid, lat_grid = np.meshgrid(gridx, gridy)
    mask = polygon_to_mask(multi_polygon, lon_grid, lat_grid)
    mask = np.where(mask == False, 1, 0)  # 生成mask，并将True/False转化为0/1
    mask_grid = np.ma.masked_array(grid, mask, fill_value=np.nan)
    mask_grid = mask_grid.filled()
    
    return mask_grid, lon_grid, lat_grid


def get_fig_ax():
    '''
    创建基础图 
    '''
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    return fig, ax


def add_scalebar(ax,lon0,lat0,length,size=0.45):
    '''
    ax: 坐标轴
    lon0: 经度
    lat0: 纬度
    length: 长度
    size: 控制粗细和距离的
    '''
    # style 3
    ax.hlines(y=lat0,  xmin = lon0, xmax = lon0+length/111, colors="black", ls="-", lw=1, label='%d km' % (length))
    ax.vlines(x = lon0, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    ax.vlines(x = lon0+length/2/111, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    ax.vlines(x = lon0+length/111, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    ax.text(lon0+length/111,lat0+size+0.01,'%d' % (length),horizontalalignment = 'center')
    ax.text(lon0+length/2/111,lat0+size+0.01,'%d' % (length/2),horizontalalignment = 'center')
    ax.text(lon0,lat0+size+0.01, '0',horizontalalignment = 'center')
    ax.text(lon0+length/111/2*3,lat0+size+0.01,'km',horizontalalignment = 'center')
    
    # style 1
    # print(help(ax.vlines))
    # ax.hlines(y=lat0,  xmin = lon0, xmax = lon0+length/111, colors="black", ls="-", lw=2, label='%d km' % (length))
    # ax.vlines(x = lon0, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=2)
    # ax.vlines(x = lon0+length/111, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=2)
    # # ax.text(lon0+length/2/111,lat0+size,'500 km',horizontalalignment = 'center')
    # ax.text(lon0+length/2/111,lat0+size,'%d' % (length/2),horizontalalignment = 'center')
    # ax.text(lon0,lat0+size,'0',horizontalalignment = 'center')
    # ax.text(lon0+length/111/2*3,lat0+size,'km',horizontalalignment = 'center')

    # style 2
    # plt.hlines(y=lat0,  xmin = lon0, xmax = lon0+length/111, colors="black", ls="-", lw=1, label='%d km' % (length))
    # plt.vlines(x = lon0, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    # plt.vlines(x = lon0+length/111, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    # plt.text(lon0+length/111,lat0+size,'%d km' % (length),horizontalalignment = 'center')
    # plt.text(lon0,lat0+size,'0',horizontalalignment = 'center')
    

def add_north(ax, labelsize=18, loc_x=0.9, loc_y=0.97, width=0.03, height=0.075, pad=0.16):
    """
    画一个比例尺带'N'文字注释
    主要参数如下
    :param ax: 要画的坐标区域 Axes实例 plt.gca()获取即可
    :param labelsize: 显示'N'文字的大小
    :param loc_x: 以文字下部为中心的占整个ax横向比例
    :param loc_y: 以文字下部为中心的占整个ax纵向比例
    :param width: 指南针占ax比例宽度
    :param height: 指南针占ax比例高度
    :param pad: 文字符号占ax比例间隙
    :return: None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen*(loc_x - width*.5), miny + ylen*(loc_y - pad)]
    right = [minx + xlen*(loc_x + width*.5), miny + ylen*(loc_y - pad)]
    top = [minx + xlen*loc_x, miny + ylen*(loc_y - pad + height)]
    center = [minx + xlen*loc_x, left[1] + (top[1] - left[1])*.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen*loc_x,
            y=miny + ylen*(loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)
    

def plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, save_path):
    '''
    根据掩膜后的网格数据，先画图后保存
    exp_name: 情景名，用于保存文件名
    insti_name: 模式名，用于保存文件名
    year_name: 年份or最大/最小/变率，用于保存文件名
    '''
    fig, ax = get_fig_ax()
    year_name = str(year_name)
    
    # 画结果网格
    mesh = ax.contourf(lon_grid, lat_grid, mask_grid, transform=ccrs.PlateCarree(), alpha=0.8, cmap='jet', extend='both')
    cbar = fig.colorbar(mesh, ax=ax, extend='neither', shrink=0.75, spacing='uniform') # 添加colorbar
    # cbar.set_label('气温 $\mathrm{degree}$', fontsize=12, loc='top')
    
    # 画边界
    shp = gpd.read_file(shp_path,encoding='utf-8')
    shp_feature = cfeat.ShapelyFeature(shp['geometry'], ccrs.PlateCarree(), edgecolor='k', facecolor='none')
    ax.add_feature(shp_feature, linewidth=0.7, alpha=0.4)

    # 添加网格线/经纬度
    grid = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.6, alpha=0.7, x_inline=False, y_inline=False, color='grey')
    grid.top_labels=False
    grid.right_labels=False
    grid.xformatter = LONGITUDE_FORMATTER
    grid.yformatter = LATITUDE_FORMATTER
    # grid.xlocator = mticker.FixedLocator(np.arange(np.floor(lon_min), np.ceil(lon_max), 1.5)) # 经纬度范围自定义
    # grid.ylocator = mticker.FixedLocator(np.arange(np.floor(lat_min), np.ceil(lat_max), 1.5))
    grid.xlabel_style={'size':13}
    grid.ylabel_style={'size':13}
    ax.set_extent([89,104,31,40.5],crs=ccrs.PlateCarree()) # 写死了，青海省
    
    # ax.set_title(time, loc='left', fontsize=20, weight='normal')     
    # ax.set_title(str(time)+' 门头沟区域积水风险结果', loc='center', fontsize=20 ,weight='normal')
    # ax.set_title('unit:'+unit, loc='right', fontsize=18 ,weight='normal')
    ax.text(0.5, 0.92, f'青海省{year_name}年要素分布图', transform=ax.transAxes, fontdict={'size':'15','color':'black'}, horizontalalignment='center')
    ax.text(0.7, 0.05, '青海省气候中心 制', transform=ax.transAxes, fontdict={'size':'15','color':'black'})
    
    # 画指南针和比例尺
    add_north(ax)
    add_scalebar(ax,90,31.5,150,size=0.1)

    # 保存图片
    save_path1 = save_path + '/{}_{}_{}_结果图.png'.format(exp_name, insti_name, year_name)
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')
    gc.collect()
    
    return save_path1
    
    
# 读取数据
# from Module02.page_climate.page_climate_handler import climate_forcast
# data_json = dict()
# data_json['time_freq'] = 'Y'
# data_json['evaluate_times'] = '1950,1980' # 预估时段时间条
# data_json['refer_years'] = '2000,2024'# 参考时段时间条
# data_json['sta_ids'] = '51886,51991,52602,52633,52645,52657,52707,52713,52737,52745,52754,52765,52818,52825,52833,52836,52842,52851,52853,52855,52856,52859,52862,52863,52866,52868,52869,52874,52875,52876,52877,52908,52942,52943,52955,52957,52963,52968,52972,52974,56004,56015,56016,56018,56021,56029,56033,56034,56043,56045,56046,56065,56067,56125,56151'
# data_json['cmip_type'] = 'original' # 预估数据类型 原始/delta降尺度/rf降尺度/pdf降尺度
# data_json['cmip_res'] = None # 分辨率 1/5/10/25/50/100 km
# data_json['cmip_model'] = ['BCC-CSM2-MR', 'CanESM5']# 模式，列表：['CanESM5','CESM2']等
# data_json['element'] = 'TEM_Avg'
# single_cmip_res, lon_list, lat_list = climate_forcast(data_json)


# save_path = r'C:/Users/MJY/Desktop/result'
# shp_path = r'C:/Users/MJY/Desktop/青海省.json'
# method = 'idw'

# all_png = dict()
# for exp, sub_dict1 in single_cmip_res.items():
#     all_png[exp] = dict()
#     for insti,stats_table in sub_dict1.items():
#         all_png[exp][insti] = dict()
#         for i in tqdm(range(len(stats_table))):
#             value_list = stats_table.iloc[i,1:-3].tolist()
#             year_name = stats_table.iloc[i,0]
#             exp_name = exp
#             insti_name = insti
#             # 插值 掩膜 画图 保存
#             mask_grid, lon_grid, lat_grid = interp_and_mask(shp_path, lon_list, lat_list, value_list, method)
#             png_path = plot_and_save(shp_path, mask_grid, lon_grid, lat_grid, exp_name, insti_name, year_name, save_path)
#             all_png[exp][insti][year_name] = png_path
        
        

    









