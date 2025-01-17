# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:38:58 2024

@author: EDY
"""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as ctk
import xeofs as xe
from Utils.config import cfg
from matplotlib.path import Path
from cartopy.io.shapereader import BasicReader
from cartopy.mpl.patch import geos_to_path
from xeofs.models import EOF, EOFRotator
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_eof_and_pc(lons, lats, eof, pc, ax1, ax2, EOF, PC, time_min, time_max, path):

    mesh = ax1.contourf(lons, lats, eof.squeeze(), cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    cb = plt.colorbar(mesh, ax=ax1, extend='both', shrink=0.9, pad=0.02)
    cb.set_label('correlation coefficient', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax1.set_title(f'{EOF}', fontsize=12, loc='left', pad=10)
    gl = ax1.gridlines(draw_labels=True, x_inline=False, y_inline=False, linestyle='dashed', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.xlocator = ctk.LongitudeLocator(10)
    gl.ylocator = ctk.LatitudeLocator(6)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    for col in mesh.collections:
        col.set_clip_path(path, ccrs.PlateCarree()._as_mpl_transform(ax1))

    years = range(int(time_min), int(time_max) + 1)
    ax2.plot(years, pc, color='b', linewidth=1.5)
    ax2.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax2.set_title(f'{PC}', fontsize=12, loc='left', pad=10)
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Normalized Units', fontsize=10)
    ax2.tick_params(axis='both', labelsize=9)
    ax2.set_xlim(int(time_min), int(time_max))
    ax2.set_xticks(years[::4])


def eof(ds, shp_name, output_filepath):
    eof = xe.models.EOF(n_modes=4)
    eof.fit(ds.data_year, dim="time")
    comps = eof.components()  # EOFs (spatial patterns)
    scores = eof.scores()  # PCs (temporal patterns)

    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 0.8])
    EOFs = ['EOF1', 'EOF2', 'EOF3', 'EOF4']
    PCs = ['PC1', 'PC2', 'PC3', 'PC4']
    lon, lat = np.meshgrid(ds.longitude, ds.latitude)
    year = ds.time

    area = BasicReader(shp_name)
    geo_list = list(area.geometries())
    path = Path.make_compound_path(*geos_to_path(geo_list))

    for i, EOF1 in enumerate(EOFs):
        ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[i, 1])
        plot_eof_and_pc(lon, lat, comps[i], scores[i, :], ax1, ax2, EOFs[i], PCs[i], year[0], year[-1], path)

    result_picture = os.path.join(output_filepath, 'EOF.png')
    fig.savefig(result_picture, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()

    result_picture = result_picture.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)
    result_picture = result_picture.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)

    return result_picture


def reof(ds, shp_name, output_filepath):
    components = []
    scores = []
    model = EOF(n_modes=4, standardize=True, use_coslat=True)
    model.fit(ds.data_year, dim="time")
    rot_var = EOFRotator(n_modes=4, power=1)
    rot_var.fit(model)
    components.append(rot_var.components())
    scores.append(rot_var.scores())

    comps = components[0]
    scores = scores[0]

    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 0.8])
    EOFs = ['EOF1', 'EOF2', 'EOF3', 'EOF4']
    PCs = ['PC1', 'PC2', 'PC3', 'PC4']
    lon, lat = np.meshgrid(ds.longitude, ds.latitude)
    year = ds.time

    area = BasicReader(shp_name)
    geo_list = list(area.geometries())
    path = Path.make_compound_path(*geos_to_path(geo_list))

    for i, EOF1 in enumerate(EOFs):
        ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[i, 1])
        plot_eof_and_pc(lon, lat, comps[i], scores[i, :], ax1, ax2, EOFs[i], PCs[i], year[0], year[-1], path)

    result_picture = os.path.join(output_filepath, 'REOF.png')
    fig.savefig(result_picture, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()

    result_picture = result_picture.replace(cfg.INFO.IN_DATA_DIR, cfg.INFO.OUT_DATA_DIR)
    result_picture = result_picture.replace(cfg.INFO.OUT_DATA_DIR, cfg.INFO.OUT_DATA_URL)

    return result_picture


if __name__ == "__main__":
    output_filepath = r'D:\Project\1'
    nc_path = r'D:\Project\1\data.nc'
    shp_name = r'D:\Project\3_项目\11_生态监测评估体系建设-气候服务系统\材料\03-边界矢量\03-边界矢量\08-省州界\省界.shp'

    ds = xr.open_dataset(nc_path)
    eof_path = eof(ds, shp_name, output_filepath)
    reof_path = reof(ds, shp_name, output_filepath)
