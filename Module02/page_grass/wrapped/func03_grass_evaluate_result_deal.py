# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:59:59 2024

@author: EDY
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:40:04 2024

@author: EDY
"""

from Utils.data_loader_with_threads import get_database_result



def grass_table_stats(element,time_freq,stats_times,sta_ids):
    
    elements= 'Datetime,Station_Id_C,Station_Name,Lon,Lat,value,type'
    sta_ids = tuple(sta_ids.split(','))

    if element=='grassland_green_period':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(11,),1,1)
        data_r_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(10,),0,0)
        data_r_df.reset_index(inplace=True)

        station_df.reset_index(inplace=True,drop=True)
        
    elif element=='grassland_yellow_period':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(21,),1,1)
        data_r_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(20,),0,0)
        data_r_df.reset_index(inplace=True)

        station_df.reset_index(inplace=True,drop=True)
        
    elif element=='grassland_growth_period':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(31,),1,1)

        station_df.reset_index(inplace=True,drop=True)
        
    elif element=='dwei':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(40,),1,1)

        station_df.reset_index(inplace=True,drop=True)
    
    elif element=='grassland_coverage':
        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(50,),1,1)

        station_df.reset_index(inplace=True,drop=True)
    
    elif element=='grass_height':

        station_df,data_df=get_database_result(sta_ids, elements, 'qh_climate_grass_growth', time_freq, stats_times,(60,),1,1)

        station_df.reset_index(inplace=True,drop=True)


    
    if element in ['grassland_green_period','grassland_yellow_period']:
        return station_df,data_r_df,data_df
    else:
        return station_df,data_df

        
        
        
    