# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:38:06 2024

@author: EDY

模式数据路径拼接

根目录/时间尺度/模式名称/情景模式/元素名称
"""
import os
import pandas as pd


def read_xlsx_data(file,station_id):
    data_1=pd.read_csv(file)
    if 'Unnamed: 0' in data_1.columns:
        data_1.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    data_1.iloc[0,1::]=data_1.iloc[0,1::].astype(int).astype(str)
    ori_columns =data_1.columns
    data_1.columns=data_1.loc[0]
    data_1=data_1.copy()
    data_1.loc[0] =ori_columns
    data_11=data_1[['station id']+station_id]    
    data_11.rename(columns={data_11.columns[0]: 'Datetime'}, inplace=True)
    data_11=data_11.iloc[3::,:]
    data_11['Datetime'] = pd.to_datetime(data_11['Datetime'])
    data_11.set_index('Datetime', inplace=True) 
    # data_11.drop(['Datetime'], axis=1, inplace=True) 

    return data_11

def read_model_data(data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_id):
    
    # 通过时间尺度，取出时间年月日：
    if time_freq== 'Y':
        # Y
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        
        if int(start_year)<2015:
            
            file_1=os.path.join(data_dir,time_scale,insti,'historical',var+'.csv')
            data_1= read_xlsx_data(file_1,station_id)
            data_m1= data_1.loc[start_year:'2014']
            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_m2= data_2.loc['2015':end_year]

            data_m3=pd.concat([data_m1,data_m2],axis=0)
            
        else:
            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_m3 =data_2.loc[start_year:end_year]
            
    elif time_freq in ['Q']:

        # Q
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        month = stats_times[1]
        month = list(map(int,month.split(',')))
        
        
        if 12 in month:
            
            if int(start_year)<2016:
                
                file_1=os.path.join(data_dir,time_scale,insti,'historical',var+'.csv')
                data_1= read_xlsx_data(file_1,station_id)            
                file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
                data_2= read_xlsx_data(file_2,station_id)
                
                data_m1 = data_1[str(int(start_year)-1)+'-12':'2014-12']
                data_m1 = data_m1[data_m1.index.month.isin(month)]

                data_m2 = data_2['2015-01':str(int(end_year)-1)+'-02']
                data_m2 = data_m2[data_m2.index.month.isin(month)]

                data_m3=pd.concat([data_m1,data_m2],axis=0)
                
            else:
                file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
                data_2= read_xlsx_data(file_2,station_id)
                
                data_m2 = data_2[str(int(start_year)-1)+'-12':str(int(end_year)-1)+'-02']
                data_m2 = data_m2[data_m2.index.month.isin(month)]

                data_m3=pd.concat([data_m1,data_m2],axis=0)
            
        else:
            
            if int(start_year)<2015:

                 file_1=os.path.join(data_dir,time_scale,insti,'historical',var+'.csv')
                 data_1= read_xlsx_data(file_1,station_id)            
                 file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
                 data_2= read_xlsx_data(file_2,station_id)
        
                 data_m1 = data_1[data_1.index.month.isin(month)]
                 data_m1 = data_m1.loc[start_year:'2014']
                
                 data_m2 = data_2[data_2.index.month.isin(month)]
                 data_m2 = data_m2.loc['2015':end_year]
       
                 data_m3=pd.concat([data_m1,data_m2],axis=0)
             
            else:
            
                file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
                data_2= read_xlsx_data(file_2,station_id)
                data_m2 = data_2[data_2.index.month.isin(month)]
                data_m3 = data_m2.loc[start_year:end_year]
                
    elif time_freq in ['M2']:

        # Q
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        month = stats_times[1]
        month = list(map(int,month.split(',')))
        
        if int(start_year)<2015:

             file_1=os.path.join(data_dir,time_scale,insti,'historical',var+'.csv')
             data_1= read_xlsx_data(file_1,station_id)            
             file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
             data_2= read_xlsx_data(file_2,station_id)
    
             data_m1 = data_1[data_1.index.month.isin(month)]
             data_m1 = data_m1.loc[start_year:'2014']
            
             data_m2 = data_2[data_2.index.month.isin(month)]
             data_m2 = data_m2.loc['2015':end_year]
   
             data_m3=pd.concat([data_m1,data_m2],axis=0)
         
        else:
        
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_m2 = data_2[data_2.index.month.isin(month)]
            data_m3 = data_m2.loc[start_year:end_year]
    
    elif time_freq== 'M1':
        
        
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        start_year=int(start_time[:4:])
        
        if int(start_year)<2015:
            
            file_1=os.path.join(data_dir,time_scale,insti,'historical',var+'.csv')
            data_1= read_xlsx_data(file_1,station_id)
            data_m1= data_1.loc[start_time[:4:]+'-'+start_time[4::]:'2014']
            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_m2= data_2.loc['2015':end_time[:4:]+'-'+end_time[4::]]
   
            data_m3=pd.concat([data_m1,data_m2],axis=0)
            
        else:
            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_m3 =data_2.loc[start_time[:4:]+'-'+start_time[4::]:end_time[:4:]+'-'+end_time[4::]]
        
    elif time_freq== 'D1':
    
        start_time = stats_times.split(',')[0]
        end_time = stats_times.split(',')[1]
        start_year=int(start_time[:4:])
        
        if int(start_year)<2015:
            
            file_1=os.path.join(data_dir,time_scale,insti,'historical',var+'.csv')
            data_1= read_xlsx_data(file_1,station_id)
            data_m1= data_1.loc[start_time[:4:]+'-'+start_time[4:6:]+'-'+start_time[6::]:'2014']
            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_m2= data_2.loc['2015':end_time[:4:]+'-'+end_time[4:6:]+'-'+end_time[6::]]
   
            data_m3=pd.concat([data_m1,data_m2],axis=0)
            
        else:
            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_m3 =data_2.loc[start_time[:4:]+'-'+start_time[4:6:]+'-'+start_time[6::]:end_time[:4:]+'-'+end_time[4::]+'-'+end_time[6::]]
    
    elif time_freq== 'D2':
        
        def read_and_merge_data(data_1,data_2, start_time, end_time, year):
            start_date = f"{year}-{start_time[:2]}-{start_time[2:]}"
            end_date = f"{year}-{end_time[:2]}-{end_time[2:]}"
            
            if year<2014:
                return data_1.loc[start_date:end_date]
            else:
                return data_2.loc[start_date:end_date]

        years = stats_times[0]
        start_year = int(years.split(',')[0])
        end_year = int(years.split(',')[1])
        date_time = stats_times[1]
        start_time = date_time.split(',')[0]
        end_time = date_time.split(',')[1]

        if int(start_year)<2015:
            file_1=os.path.join(data_dir,time_scale,insti,'historical',var+'.csv')
            data_1= read_xlsx_data(file_1,station_id)            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
               
        else:
            
            file_2=os.path.join(data_dir,time_scale,insti,scene,var+'.csv')
            data_2= read_xlsx_data(file_2,station_id)
            data_1=[]
            
        data_m3 = pd.concat([read_and_merge_data(data_1,data_2, start_time, end_time, i) for i in range(start_year, end_year + 1)], axis=0)


     
    return data_m3
        
        

if __name__=='__main__':
    
    data_dir=r'D:\Project\qh'
    time_scale= 'daily'
    insti='Set'
    scene='ssp126'
    var='tas'
    station_id=['51886','52602']
    
    stats_times='2030,2040'
    time_freq= 'Y'
    
    stats_times=['2011,2040','12,1,2']
    time_freq= 'Q'
    
    stats_times='201102,202005'
    time_freq= 'M1'
    
    stats_times='20110205,20200505'
    time_freq= 'D1'
    
    stats_times=['2011,2040','0505,0805']
    time_freq= 'D2'
    
    time_freq = 'M2'
    stats_times = ["2010,2025", "1,2"]

    data=read_model_data(data_dir,time_scale,insti,scene,var,stats_times,time_freq,station_id)