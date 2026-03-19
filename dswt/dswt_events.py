import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from readers.read_dswt_output import read_df_from_multiple_csvs
from tools.timeseries import get_l_time_range
from tools.files import get_dir_from_json

from scipy.signal import find_peaks
from datetime import datetime
import numpy as np
import pandas as pd


class DswtEvent:
    def __init__(self,
                 start_time:datetime,
                 end_time:datetime,
                 duration:int,
                 f_dswt:float,
                 max_vel:float,
                 mean_vel:float,
                 std_vel:float,
                 max_thickness:float,
                 mean_thickness:float,
                 std_thickness:float,
                 max_h:float,
                 mean_h:float,
                 std_h:float,
                 max_transport:float,
                 mean_transport:float,
                 std_transport:float):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.f_dswt = f_dswt
        self.max_vel = max_vel
        self.mean_vel = mean_vel
        self.std_vel = std_vel
        self.max_thickness = max_thickness
        self.mean_thickness = mean_thickness
        self.std_thickness = std_thickness
        self.max_h = max_h
        self.mean_h = mean_h
        self.std_h = std_h
        self.max_transport = max_transport
        self.mean_transport = mean_transport
        self.std_transport = std_transport
      
class DswtEvents:
    def __init__(self,
                 time:np.ndarray[datetime],
                 n_events:np.ndarray[int],
                 events:list[list[DswtEvent]]):
        self.time = time
        self.n_events = n_events
        self.events = events
    
    @staticmethod
    def calculate_from_df_timeseries(df_timeseries_all:pd.DataFrame, years:list[int], depth_contour=50, req_months=np.arange(1, 13, 1)):
        
        time = []
        n_events = []
        events = []
        
        df_time_all = np.array([pd.to_datetime(t) for t in df_timeseries_all['time'].values])
        # filter timeseries to requested months only
        df_months = np.array([t.month for t in df_time_all])
        l_months = np.array([m in req_months for m in df_months])
        df_timeseries = df_timeseries_all.loc[l_months]
        df_time = df_time_all[l_months]
        
        for i in range(len(years)):
            l_year = [t.year == years[i] for t in df_time]
            
            df_year = df_timeseries.loc[l_year]
            df_time_year = df_time[l_year]
            
            # determine events
            yearly_events = []
            
            z = df_year[f'transport_{depth_contour}m'].values / (24*60*60)
            i_peaks, properties = find_peaks(z, height=0.05, width=(1, 20)) # width specifies minimum and maximum width
            i_left = np.floor(properties['left_ips']).astype(int)
            i_right = np.ceil(properties['right_ips']).astype(int)
            
            n = len(i_peaks)
            
            for j in range(n):
                start_event = df_time_year[i_left[j]]
                end_event = df_time_year[i_right[j]]
                duration_event = (end_event-start_event).days
                
                l_time_event = get_l_time_range(df_time_year, start_event, end_event)
                mean_f = np.nanmean(df_year['f_dswt'][l_time_event].values)
                max_vel_event = np.nanmax(df_year['vel'][l_time_event].values)
                mean_vel_event = np.nanmean(df_year['vel'][l_time_event].values)
                std_vel_event = np.nanstd(df_year['vel'][l_time_event].values)
                mean_thickness_event = np.nanmean(df_year['thickness'][l_time_event].values)
                max_thickness_event = np.nanmax(df_year['thickness'][l_time_event].values)
                std_thickness_event = np.nanstd(df_year['thickness'][l_time_event].values)
                max_h_event = np.nanmean(df_year['mean_h'][l_time_event].values)
                mean_h_event = np.nanmax(df_year['mean_h'][l_time_event].values)
                std_h_event = np.nanstd(df_year['mean_h'][l_time_event].values)
                max_transport_event = np.nanmax(df_year[f'transport_{depth_contour}m'][l_time_event].values)
                mean_transport_event = np.nanmean(df_year[f'transport_{depth_contour}m'][l_time_event].values)
                std_transport_event = np.nanstd(df_year[f'transport_{depth_contour}m'][l_time_event].values)
                
                yearly_events.append(DswtEvent(start_event, end_event, duration_event, mean_f,
                                               max_vel_event, mean_vel_event, std_vel_event,
                                               max_thickness_event, mean_thickness_event, std_thickness_event,
                                               max_h_event, mean_h_event, std_h_event,
                                               max_transport_event, mean_transport_event, std_transport_event))
                
            time.append(datetime(years[i], 7, 17))
            n_events.append(n)
            events.append(yearly_events)
        
        return DswtEvents(np.array(time), np.array(n_events), events)
    
if __name__ == '__main__':
    years = np.arange(2017, 2019)
    input_dir = f'{get_dir_from_json("output")}processed/'
    df_timeseries = read_df_from_multiple_csvs(input_dir, years, 'dswt_timeseries_')
    dswt_events = DswtEvents.calculate_from_df_timeseries(df_timeseries, years, req_months=[5, 6, 7])
