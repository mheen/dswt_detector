from ocean_model_data import select_input_files, load_roms_data, select_roms_subset
from tools.dswt_detection import determine_dswt_along_multiple_transects
from tools import log
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def write_daily_mean_dswt_fraction_to_csv(input_dir:str, files_contain:str, grid_file:str,
                                          transects_file:str, output_file:str,
                                          lon_range=None, lat_range=None):
    
    roms_files = select_input_files(input_dir, files_contain)
    roms_files.sort()

    time = []
    f_dswt = []
    for file in roms_files:
        # --- Load ROMS data
        ds = load_roms_data(file, grid_file)
        
        if lon_range is not None and lat_range is not None: # does this make computation faster?
            ds = select_roms_subset(ds, time_range=None, lon_range=lon_range, lat_range=lat_range)
            
        # --- Find DSWT along transects
        l_dswt = determine_dswt_along_multiple_transects(ds, transects_file)    
        
        # get daily mean percentage of DSWT occurrence along transects
        # !!! FIX !!! assuming here that each file contains daily data -> keep?
        f_dswt.append(np.nanmean(np.sum(l_dswt, axis=1)/l_dswt.shape[1]))
        ocean_time0 = pd.to_datetime(ds.ocean_time.values[0])
        time.append(datetime(ocean_time0.year, ocean_time0.month, ocean_time0.day))

    # --- Write to output file
    log.info(f'Writing daily fraction DSWT occurrence to file: {output_file}')
    time = np.array(time).flatten()
    f_dswt = np.array(f_dswt).flatten()
    df = pd.DataFrame(np.array([time, f_dswt]).transpose(), columns=['time', 'f_dswt'])
    df.to_csv(output_file, index=False)

def calculate_monthly_mean_dswt_fraction(input_path:str) -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    df = pd.read_csv(input_path)
    time = pd.to_datetime(df['time'].values)
    f_dswt = df['f_dswt'].values
    
    time_m = []
    f_dswt_m = []
    for n in range(time[0].month, time[-1].month+1):
        l_time = [t.month == n for t in time]
        time_m.append(datetime(time[0].year, n, 1))
        f_dswt_m.append(np.nanmean(f_dswt[l_time]))
    
    time_m = np.array(time_m)
    f_dswt_m = np.array(f_dswt_m)
    
    return time_m, f_dswt_m