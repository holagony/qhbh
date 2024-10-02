import os
import numpy as np
import xarray as xr
from osgeo import gdal
from datetime import datetime
from Utils.config import cfg


def tif_dataloader(tif_path, key='dem'):
    dataset = gdal.Open(tif_path)
    assert dataset is not None, 'tif文件不存在'

    nXSize = dataset.RasterXSize  # 列 col 对应经度
    nYSize = dataset.RasterYSize  # 行 row 对应纬度
    bands = dataset.RasterCount  # 波段
    adfGeoTransform = dataset.GetGeoTransform()  # 仿射矩阵

    # 生成经纬度网格 对应np.meshgrid
    lon_lat = []

    if key == 'dem':
        for i in range(nYSize):
            row = []
            for j in range(nXSize):
                px = adfGeoTransform[0] + j * adfGeoTransform[1] + i * adfGeoTransform[2]
                py = adfGeoTransform[3] + j * adfGeoTransform[4] + i * adfGeoTransform[5]
                col = [px, py]
                row.append(col)

            lon_lat.append(row)

        lon_lat = np.array(lon_lat)
        lon_lat = lon_lat.transpose(2, 0, 1)

    else:
        lon_lat = np.nan

    for i in range(bands):
        data = dataset.GetRasterBand(i + 1).ReadAsArray()
        data = np.expand_dims(data, 0)

        if i == 0:
            all_data = data
        else:
            all_data = np.concatenate((all_data, data), axis=0)

    all_data = np.where(all_data == 32767, np.nan, all_data)
    all_data = all_data[0]

    # 结果输出为dict
    data = {'data': all_data, 'lon_lat': lon_lat, 'transform': adfGeoTransform, 'projection': dataset.GetProjection(), 'bands': bands, 'col': nXSize, 'row': nYSize}

    return data


def array2nc(data, lat, lon, var_name, time=None, height=None, interp=cfg.INFO.PRODUCT_RESIZE):

    if len(data.shape) == 4:
        da = xr.DataArray(data, coords=[time, height, lat, lon], dims=['time', 'height', 'lat', 'lon'])
        ds = xr.Dataset({var_name: da})
        ds['lon'].attrs['units'] = "degrees_east"
        ds['lon'].attrs['long_name'] = "Longitude"
        ds['lat'].attrs['units'] = "degrees_north"
        ds['lat'].attrs['long_name'] = "Latitude"
        ds['time'].attrs['long_name'] = "Time(CST)"
        ds['height'].attrs['units'] = "Meters"
        ds['height'].attrs['long_name'] = "Height"

    elif len(data.shape) == 3:
        if time is not None:
            da = xr.DataArray(data, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
            ds = xr.Dataset({var_name: da})
            ds['lon'].attrs['units'] = "degrees_east"
            ds['lon'].attrs['long_name'] = "Longitude"
            ds['lat'].attrs['units'] = "degrees_north"
            ds['lat'].attrs['long_name'] = "Latitude"
            ds['time'].attrs['long_name'] = "Time(CST)"

        elif height is not None:
            da = xr.DataArray(data, coords=[height, lat, lon], dims=['height', 'lat', 'lon'])
            ds = xr.Dataset({var_name: da})
            ds['lon'].attrs['units'] = "degrees_east"
            ds['lon'].attrs['long_name'] = "Longitude"
            ds['lat'].attrs['units'] = "degrees_north"
            ds['lat'].attrs['long_name'] = "Latitude"
            ds['height'].attrs['units'] = "Meters"
            ds['height'].attrs['long_name'] = "Height"

    else:
        da = xr.DataArray(data, coords=[lat, lon], dims=['lat', 'lon'])
        ds = xr.Dataset({var_name: da})
        ds['lon'].attrs['units'] = "degrees_east"
        ds['lon'].attrs['long_name'] = "Longitude"
        ds['lat'].attrs['units'] = "degrees_north"
        ds['lat'].attrs['long_name'] = "Latitude"

    units = 'dimensionless'

    if var_name == 'FLOOD':
        units = 'cm'

    elif var_name == 'PRE':
        units = 'mm'

    ds[var_name].attrs['units'] = units
    ds.attrs = dict(CreatedTime=datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))

    if len(data.shape) == 3 and interp:
        # new_lon = np.linspace(ds.lon[0], ds.lon[-1], int(ds.dims['lon']/2))
        # new_lat = np.linspace(ds.lat[0], ds.lat[-1], int(ds.dims['lat']/2))
        new_lon = np.arange(ds.lon[0], ds.lon[-1]+0.001, 0.001).round(2)
        new_lat = np.arange(ds.lat[0], ds.lat[-1]+0.001, 0.001).round(2)
        ds = ds.interp(lat=new_lat, lon=new_lon)

    return ds


def nc_save(ds, save_path, filename):

    var_name = list(ds.data_vars.keys())
    set_list = []
    os.makedirs(save_path, exist_ok=True)
    for var in var_name:
        param = {
            'complevel': 9,
            'zlib': True,
            '_FillValue': -9999,
        }
        set_list.append([var, param])

    ds.to_netcdf(os.path.join(save_path, filename), engine='netcdf4', encoding={sn: comp for [sn, comp] in set_list})
    return os.path.join(save_path, filename)


def calc_scale_and_offset(min_v, max_v, n=16):
    '''
    stretch/compress data to the avaiable packed range
    '''
    if max_v - min_v == 0:
        scale_factor = 1.0
        add_offset = 0.0
    else:
        scale_factor = (max_v - min_v) / (2**n - 1)
        # translate the range to be symmetric about zero
        add_offset = min_v + 2**(n - 1) * scale_factor

    return scale_factor, add_offset