# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:29:14 2024

@author: MJY
"""

import xarray as xr
import os
import glob
import pandas as pd

path = r'C:/Users/MJY/Desktop/data'
total_path = glob.glob(os.path.join(path,'*.nc'))


ds_list = []
for p in total_path:
    ds = xr.open_dataset(p)
    ds_list.append(ds)
    
ds_all = xr.concat(ds_list, dim='time')
datetimeindex = ds_all.indexes['time'].to_datetimeindex()
ds_all['time'] = datetimeindex.normalize()

# cc = ds_all.sel(time=slice("2003-01-01", "2003-08-02"))
# cc = ds_all.sel(time=slice("2001", "2004"))
# cc = ds_all.sel(time=slice("2003-01", "2003-08"))

# selected_data = ds_all.sel(time=datetimeindex[0:500])

# In[]
# 生成时间列表 datetimeindex
c1 = pd.date_range('2020','2021', freq='D')[:-1] # 'Y'

dates = pd.date_range(start='2015', end='2020', freq='D')[:-1]  # 'Q' or 'M2'
c2 = dates[dates.month.isin([2,3,4])]


c4 = pd.date_range(start='20150105', end='20200525', freq='D') # D1

# 生成每年的1月5号到7月13号的日期 D2
start_date = '2015-01-05'
end_date = '2020-07-13'
dates = pd.date_range(start='2015', end='2020', freq='D')
c6 = dates[((dates.month==1) & (dates.day>=5)) | ((dates.month>1) & (dates.month<7)) | ((dates.month==7) & (dates.day<=13))]
# 筛选出每年的1月5号到7月13号的日期




# In[]
s = '201501'
e = '202005'

s = pd.to_datetime(s,format='%Y%m')
e = pd.to_datetime(e,format='%Y%m') + pd.DateOffset(months=1)


c3 = pd.date_range(start=s, end=e, freq='D')[:-1] # M1


