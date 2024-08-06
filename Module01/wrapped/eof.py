# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:38:58 2024

@author: EDY
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from eofs.standard import Eof
import cartopy.mpl.ticker  as ctk

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 



def plot_eof_and_pc(lons, lats, eof, pc, var,  ax1, ax2,EOF,PC,time_min,time_max):

    
    mesh = ax1.contourf(lons, lats, eof.squeeze(),cmap=plt.cm.RdBu_r,transform=ccrs.PlateCarree())
    cb = plt.colorbar(mesh, ax=ax1, extend='both', shrink=0.8)
    cb.set_label('correlation coefficient', fontsize=12)
    ax1.set_title(f'{EOF}  ', fontsize=16,loc='left')
    gl = ax1.gridlines(draw_labels=True, x_inline=False, y_inline=False, linestyle='dashed')
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.xlocator = ctk.LongitudeLocator(20)
    gl.ylocator = ctk.LatitudeLocator(8)
    gl.xformatter = ctk.LongitudeFormatter(zero_direction_label=True)
    gl.yformatter = ctk.LatitudeFormatter()

    years = range(int(time_min),int(time_max)+1)
    ax2.plot(years, pc, color='b', linewidth=2)
    ax2.axhline(0, color='k')
    ax2.set_title(f'{PC}  ',loc='left')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Normalized Units')
    ax2.set_xlim(int(time_min),int(time_max))
    # ax2.set_ylim(-3, 3)
    ax2.set_title(f'Var={var:.2}', loc='right')

def eof(data,gridx,gridy,year):
    coslat = np.cos(np.deg2rad(gridy))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    
    # 计算EOF & PC
    solver = Eof(data, weights=wgts)
    eof = solver.eofsAsCorrelation(neofs=4)
    pc = solver.pcs(npcs=4, pcscaling=1)
    var = solver.varianceFraction(neigs=4)
    
    fig = plt.figure(figsize=(10, 10))
    
    EOFs = ['EOF1', 'EOF2','EOF3','EOF4',]
    PCs = ['PC1', 'PC2','PC3','PC4']
    lon,lat=np.meshgrid(gridx,gridy)
    
    
    for i, EOF in enumerate(EOFs):
        
        print(i,EOF,)
        # 第一个子图带投影
        ax1 = fig.add_subplot(4, 2, 2*i+1, projection=ccrs.PlateCarree())
        # 第二个子图不带投影
        ax2 = fig.add_subplot(4, 2, 2*i+2)
        plot_eof_and_pc(lon, lat, eof[i], pc[:,i], var[i], ax1, ax2,EOFs[i],PCs[i],year[0],year[-1])
        
    plt.tight_layout()
    plt.show()