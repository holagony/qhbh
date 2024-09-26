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
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry.multipolygon import MultiPolygon
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeat
from Utils.station_to_grid import station_to_grid
from shapely.geometry import  Polygon
from Utils.config import cfg
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 中文字体可修改
mpl.rcParams['axes.unicode_minus'] = False



    
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
    shp = gpd.read_file(shp_path,encoding='gbk')
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


def add_scalebar(ax, x0, y0, length, size=0.014):
    '''
    ax: 坐标轴
    x0: 比例尺起点的x坐标（0-1之间）
    y0: 比例尺起点的y坐标（0-1之间）
    length: 比例尺的长度（单位：km）
    size: 控制粗细和距离的
    '''
    # 获取当前坐标轴的范围
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 计算比例尺的实际长度（根据坐标轴的范围）
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # 将0-1坐标系转换为实际坐标系
    x0_actual = xlim[0] + x0 * x_range
    y0_actual = ylim[0] + y0 * y_range
    
    # 计算比例尺的实际长度（单位：度）
    length_deg = length / 111  # 1度大约等于111公里
    
    # 绘制比例尺
    ax.hlines(y=y0_actual, xmin=x0_actual, xmax=x0_actual + length_deg, colors="black", ls="-", lw=1, label='%d km' % (length))
    
    # 绘制竖线（只在上半部分）
    ax.vlines(x=x0_actual, ymin=y0_actual, ymax=y0_actual + size * y_range, colors="black", ls="-", lw=1)
    ax.vlines(x=x0_actual + length_deg / 2, ymin=y0_actual, ymax=y0_actual + size * y_range, colors="black", ls="-", lw=1)
    ax.vlines(x=x0_actual + length_deg, ymin=y0_actual, ymax=y0_actual + size * y_range, colors="black", ls="-", lw=1)
    
    # 添加文本
    ax.text(x0_actual + length_deg, y0_actual + size * y_range + 0.01 * y_range, '%d' % (length), horizontalalignment='center')
    ax.text(x0_actual + length_deg / 2, y0_actual + size * y_range + 0.01 * y_range, '%d' % (length / 2), horizontalalignment='center')
    ax.text(x0_actual, y0_actual + size * y_range + 0.01 * y_range, '0', horizontalalignment='center')
    ax.text(x0_actual + length_deg / 2 * 2.5, y0_actual + size * y_range + 0.01 * y_range, 'km', horizontalalignment='center')
     
def add_north(ax, labelsize=18, loc_x=0.9, loc_y=0.99, width=0.03, height=0.075, pad=0.16):
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
    
    lon_min=89
    lon_max=104
    lat_min=31
    lat_max=40
    lakes_shp=cfg.FILES.LAKE
    glaciers_shp=cfg.FILES.ICE
    
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())    
    # 画结果网格
    mesh = ax.contourf(lon_grid, lat_grid, mask_grid, transform=ccrs.PlateCarree(), alpha=0.8, cmap='jet', extend='both')
    
    # 画边界
    shp = gpd.read_file(shp_path,encoding='utf-8')
    shp_feature = cfeat.ShapelyFeature(shp['geometry'], ccrs.PlateCarree(), edgecolor='k', facecolor='none')
    ax.add_feature(shp_feature, linewidth=0.7, alpha=0.4)
    
    # 合并所有多边形
    try:
        merged_geometry = shp.geometry.unary_union
        
        # 提取最外部的边界
        if isinstance(merged_geometry, MultiPolygon):
            exterior_boundary = MultiPolygon([Polygon(geom.exterior) for geom in merged_geometry.geoms])
        elif isinstance(merged_geometry, Polygon):
            exterior_boundary = Polygon(merged_geometry.exterior)
        else:
            raise ValueError("Unexpected geometry type")
        
        # 创建外部边界的 ShapelyFeature
        exterior_feature = cfeat.ShapelyFeature(exterior_boundary, ccrs.PlateCarree(), edgecolor='k', facecolor='none')
        ax.add_feature(exterior_feature, linewidth=1.0, alpha=0.7)
    except:
        print("Unexpected geometry type")


    # 湖泊、冰川
    lakes_gdf = gpd.read_file(lakes_shp)
    glaciers_gdf = gpd.read_file(glaciers_shp)
    lakes_gdf.plot(ax=ax, color='blue', label='湖泊')
    glaciers_gdf.plot(ax=ax, color='#73ffdf', label='冰川')

    # 添加网格线/经纬度
    grid = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.6, alpha=0.7, x_inline=False, y_inline=False, color='grey')
    grid.top_labels=False
    grid.right_labels=False
    grid.xformatter = LONGITUDE_FORMATTER
    grid.yformatter = LATITUDE_FORMATTER

    grid.xlabel_style={'size':13}
    grid.ylabel_style={'size':13}
    ax.set_extent([lon_min,lon_max,lat_min,lat_max],crs=ccrs.PlateCarree()) 
    
    # ax.text(0.5, 0.96, f'青海省{year_name}要素分布图', transform=ax.transAxes, fontdict={'size':'15','color':'black'}, horizontalalignment='center')
    ax.text(0.8, 0.02, '青海省气候中心', transform=ax.transAxes, fontdict={'size':'10','color':'black'})
    
    # 画指南针和比例尺
    add_north(ax)
    add_scalebar(ax,0.8, 0.05,200,size=0.014)

    lakes_handle = mpatches.Rectangle((0, 0), 1, 1, facecolor='blue', label='湖泊')
    glaciers_handle = mpatches.Rectangle((0, 0), 1, 1, facecolor='#73ffdf', label='冰川')
    state_handle = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=0.7,alpha=0.4, label='州界')
    province_handle = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=0.7,alpha=1, label='省界')

    # 添加图例
    legend = ax.legend(handles=[province_handle,lakes_handle,state_handle,  glaciers_handle], 
                   loc='lower left', 
                   fontsize=10, 
                   ncol=2,  
                   frameon=False, 
                   title='图例',  
                   title_fontsize=10, 
                   handletextpad=0.5,  
                   columnspacing=1.0,
                   bbox_to_anchor=(0.0, 0.05))  

    # 手动调整图例框的位置
    legend._legend_box.align = 'left'
    
    cax = ax.inset_axes([0.02, 0.035, 0.4, 0.02]) 
    cbar = fig.colorbar(mesh, cax=cax,orientation='horizontal',shrink=0.01, spacing='uniform',extend='none')
    cbar.ax.tick_params(labelsize=7)  

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
        
        

    









