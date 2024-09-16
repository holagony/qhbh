# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:51:00 2024

@author: EDY

从数据库截取数据


"""
import pandas as pd
import psycopg2
from psycopg2 import sql
from Utils.config import cfg
import numpy as np

def data_read_sql(sta_ids,elements,stats_times,table_name,time_freq):
    # 从数据库截数据
    conn = psycopg2.connect(database=cfg.INFO.DB_NAME, user=cfg.INFO.DB_USER, password=cfg.INFO.DB_PWD, host=cfg.INFO.DB_HOST, port=cfg.INFO.DB_PORT)
    cur = conn.cursor()
    sta_ids = tuple(sta_ids.split(','))

    if time_freq == 'Y':  # '%Y,%Y'
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND station_id_c IN %s
                        """)

        # 根据sql获取统计年份data
        start_year = stats_times.split(',')[0]
        end_year = stats_times.split(',')[1]
        cur.execute(query, (start_year, end_year, sta_ids))
        data = cur.fetchall()

    elif time_freq == 'Q':  # ['%Y,%Y','3,4,5']

        mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
        mon_ = tuple(mon_list)
        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]

        if 12 in mon_list:
            
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                                WHERE (SUBSTRING(datetime, 1, 4) BETWEEN %s AND %s) 
                                AND SUBSTRING(datetime, 6, 2) IN ('12', '01', '02')
                                OR (SUBSTRING(datetime, 1, 4) = %s AND SUBSTRING(datetime, 6, 2) = '12')
                                OR (SUBSTRING(datetime, 1, 4) = %s AND SUBSTRING(datetime, 6, 2) IN ('01', '02'))
                                AND station_id_c IN %s
                            """)
            cur.execute(query, (start_year, end_year,str(int(start_year)-1),str(int(end_year)+1), sta_ids))

        else:    
            query = sql.SQL(f"""
                            SELECT {elements}
                            FROM public.{table_name}
                            WHERE
                                (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                                AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                                AND station_id_c IN %s
                            """)  
            cur.execute(query, (start_year, end_year, mon_, sta_ids))

        data = cur.fetchall()

    elif time_freq == 'M1':  # '%Y%m,%Y%m'
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s))
                            AND station_id_c IN %s
                        """)

        start_year = stats_times.split(',')[0][:4]
        end_year = stats_times.split(',')[1][:4]
        start_month = stats_times.split(',')[0][4:]
        end_month = stats_times.split(',')[1][4:]
        cur.execute(query, (start_year, start_month, start_year, end_year, end_year, end_month, sta_ids))
        data = cur.fetchall()
        
    elif time_freq == 'M2':  # ['%Y,%Y','11,12,1,2']
        mon_list = [int(mon_) for mon_ in stats_times[1].split(',')]  # 提取月份
        mon_ = tuple(mon_list)
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) IN %s)
                            AND station_id_c IN %s
                        """)

        years = stats_times[0]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        cur.execute(query, (start_year, end_year, mon_, sta_ids))
        data = cur.fetchall()

    elif time_freq == 'D1':  # '%Y%m%d,%Y%m%d'
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            ((CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) >= %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) < %s)
                            OR (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) <= %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s))
                            AND station_id_c IN %s
                        """)

        start_year = stats_times.split(',')[0][:4]
        end_year = stats_times.split(',')[1][:4]
        start_month = stats_times.split(',')[0][4:6]
        end_month = stats_times.split(',')[1][4:6]
        start_date = stats_times.split(',')[0][6:]
        end_date = stats_times.split(',')[1][6:]
        cur.execute(query, (start_year, start_month, start_date, start_year, end_year, end_year, end_month, end_date, sta_ids))
        data = cur.fetchall()

    elif time_freq == 'D2':  # ['%Y,%Y','%m%d,%m%d']
        query = sql.SQL(f"""
                        SELECT {elements}
                        FROM public.{table_name}
                        WHERE
                            (CAST(SUBSTRING(datetime FROM 1 FOR 4) AS INT) BETWEEN %s AND %s
                            AND (
                                (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) >= %s)
                                OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) > %s AND CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) < %s)
                                OR (CAST(SUBSTRING(datetime FROM 6 FOR 2) AS INT) = %s AND CAST(SUBSTRING(datetime FROM 9 FOR 2) AS INT) <= %s)
                            ))
                            AND station_id_c IN %s
                        """)

        years = stats_times[0]
        dates = stats_times[1]
        start_year = years.split(',')[0]
        end_year = years.split(',')[1]
        start_mon = dates.split(',')[0][:2]
        end_mon = dates.split(',')[1][:2]
        start_date = dates.split(',')[0][2:]
        end_date = dates.split(',')[1][2:]
        cur.execute(query, (start_year, end_year, start_mon, start_date, start_mon, end_mon, end_mon, end_date, sta_ids))
        data = cur.fetchall()
        
    # 统计年份数据处理为df
    data_df = pd.DataFrame(data)
    data_df.columns = elements.split(',')
    
    for name in elements.split(',')[3::]:
        data_df.loc[data_df[name] > 1000, [name]] = np.nan
    

    return data_df