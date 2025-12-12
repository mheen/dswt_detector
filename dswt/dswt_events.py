import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

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
                 std_h:float):
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
      
class DswtEvents:
    def __init__(self,
                 time:np.ndarray[datetime],
                 n_events:np.ndarray[int],
                 events:list[list[DswtEvent]]):
        self.time = time
        self.n_events = n_events
        self.events = events
    
    @staticmethod
    def read_from_multiple_csv_files(input_dir:str, years:list):
        
        time = []
        n_events = []
        events = []
        
        for i in range(len(years)):
            input_path = f'{input_dir}dswt_{years[i]}.csv'
            if not os.path.exists(input_path):
                continue
            
            df = pd.read_csv(input_path)
            
            df['time'] = pd.to_datetime(df['time'])
            
            # determine events
            yearly_events = []
            
            df_timeseries = df.groupby(['time']).agg(
                f_dswt=('f_dswt', 'mean'),
                mean_vel=('vel', 'mean'),
                max_vel=('vel', 'max'),
                std_vel=('vel', 'std'),
                mean_h=('h', 'mean'),
                max_h=('h', 'max'),
                std_h=('h', 'std'),
                mean_thickness=('thickness', 'mean'),
                max_thickness=('thickness', 'max'),
                std_thickness=('thickness', 'std')
                )
            
            z = df_timeseries['f_dswt'].values
            i_peaks, _ = find_peaks(z, height=0.2)
            i_troughs, _ = find_peaks(1-z)
            
            n = len(i_peaks)
            
            for j in range(n):
                i_troughs_min_peak = i_troughs-i_peaks[j]
                l_subzero = i_troughs_min_peak < 0
                l_abovezero = i_troughs_min_peak > 0
                if np.sum(l_subzero) > 0:
                    i_left = i_troughs[np.where(i_troughs_min_peak == np.nanmax(i_troughs_min_peak[l_subzero]))[0][0]]
                else:
                    i_left = i_peaks[j]-1
                if np.sum(l_abovezero) > 0:
                    i_right = i_troughs[np.where(i_troughs_min_peak == np.nanmin(i_troughs_min_peak[l_abovezero]))[0][0]]
                else:
                    i_right = i_peaks[j]+1

                start_event = df_timeseries.index[i_left]
                end_event = df_timeseries.index[i_right]
                duration_event = (end_event-start_event).days
                
                l_time_event = get_l_time_range(df_timeseries.index, start_event, end_event)
                mean_f = np.nanmean(df_timeseries['f_dswt'][l_time_event].values)
                max_vel_event = np.nanmean(df_timeseries['max_vel'][l_time_event].values)
                mean_vel_event = np.nanmean(df_timeseries['mean_vel'][l_time_event].values)
                std_vel_event = np.nanmean(df_timeseries['std_vel'][l_time_event].values)
                mean_thickness_event = np.nanmean(df_timeseries['mean_thickness'][l_time_event].values)
                max_thickness_event = np.nanmean(df_timeseries['max_thickness'][l_time_event].values)
                std_thickness_event = np.nanmean(df_timeseries['std_thickness'][l_time_event].values)
                max_h_event = np.nanmean(df_timeseries['max_h'][l_time_event].values)
                mean_h_event = np.nanmean(df_timeseries['mean_h'][l_time_event].values)
                std_h_event = np.nanmean(df_timeseries['std_h'][l_time_event].values)
                
                yearly_events.append(DswtEvent(start_event, end_event, duration_event, mean_f,
                                               max_vel_event, mean_vel_event, std_vel_event,
                                               max_thickness_event, mean_thickness_event, std_thickness_event,
                                               max_h_event, mean_h_event, std_h_event))
                
            time.append(datetime(years[i], 7, 17))
            n_events.append(n)
            events.append(yearly_events)
        
                
        return DswtEvents(np.array(time), np.array(n_events), events)
    
if __name__ == '__main__':
    input_dir = f'{get_dir_from_json("output")}'
    years = np.arange(2017, 2018)
    dswt_events = DswtEvents.read_from_multiple_csv_files(input_dir, years)
