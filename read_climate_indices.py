import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from tools.timeseries import convert_time_to_datetime, get_l_time_range, add_month_to_time
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import numpy as np

# MEI: Multivariate El Nino Southern Oscillation Index
# (from NOAA https://www.psl.noaa.gov/enso/mei)
def read_mei_data(input_path=get_dir_from_json('mei'), skiprows=3) -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    df = pd.read_csv(input_path, skiprows=skiprows)
    month_strs = ['DJ', 'JF', 'FM', 'MA', 'AM', 'MJ', 'JJ', 'JA', 'AS', 'SO', 'ON', 'ND']

    time = []
    mei = []

    for year in df['YEAR'].values:
        df_year = df.loc[df['YEAR']==year]
        for m in range(1, 13):
            time.append(datetime(year, m, 15))
            mei.append(df_year[month_strs[m-1]].values[0])

    return np.array(time), np.array(mei)

# DMI: Dipole Mode Index, indicator for the Indian Ocean Dipole
# (from NOAA/ESRL: https://stateoftheocean.osmc.noaa.gov/sur/ind/dmi.php)
def read_dmi_data(input_path=get_dir_from_json('dmi')) -> tuple[np.ndarray[datetime], np.ndarray[float]]:
    
    ds = xr.load_dataset(input_path)
    time = pd.to_datetime(ds['WEDCEN2'].values)
    dmi = ds.DMI.values

    dmi_monthly_mean = []
    time_monthly_mean = np.unique([datetime(t.year, t.month, 15) for t in time])

    for t in time_monthly_mean:
        l_time = get_l_time_range(time, t, add_month_to_time(t, 1)-timedelta(days=1))
        dmi_monthly_mean.append(np.nanmean(dmi[l_time]))

    return np.array(time_monthly_mean), np.array(dmi_monthly_mean)