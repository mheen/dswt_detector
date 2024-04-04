import numpy as np
from datetime import datetime, timedelta
import re

def convert_decimal_year_to_datetime(decimal_years:np.ndarray[float]) -> np.ndarray[datetime]:
    time = []
    for d in decimal_years:
        year = int(d)
        rem = d-year
        
        base = datetime(year, 1, 1)
        time.append(base+timedelta(seconds=(base.replace(year=year+1)-base).total_seconds()*rem))
        
    return np.array(time)

def get_time_index(time_array:np.ndarray, time:datetime) -> int:
    '''Returns exact index of a requested time, raises
    error if this does not exist.'''
    t = np.where(time_array==time)[0]
    if len(t) > 1:
        raise ValueError('Multiple times found in time array that equal requested time.')
    elif len(t) == 0:
        raise ValueError('Requested time not found in time array.')
    else:
        return t[0]

def get_closest_time_index(time_array:np.ndarray, time:datetime) -> int:
    '''Returns exact index of a requested time if is exists,
    otherwise returns the index of the closest time.'''
    dt = abs(time_array-time)
    i_closest = np.where(dt == dt.min())[0][0]
    return i_closest

def get_l_time_range(time:np.ndarray, start_time:datetime, end_time:datetime) -> np.ndarray:
    if type(start_time) is datetime.date:
        start_time = datetime.datetime(start_time.year,start_time.month,start_time.day)
    if type(end_time) is datetime.date:
        end_time = datetime.datetime(end_time.year,end_time.month,end_time.day)
    l_start = time >= start_time
    l_end = time <= end_time
    l_time = l_start & l_end
    return l_time

def add_month_to_time(timestamp:datetime, n_month:int) -> datetime:
    month = timestamp.month - 1 + n_month
    year = timestamp.year + month // 12
    month = month % 12 + 1
    return datetime(year, month, timestamp.day)

def get_daily_means(time:np.ndarray, values:np.ndarray, time_axis=0) -> tuple:
    daily_time = []
    daily_values = []

    n_days = (time[-1]-time[0]).days+1

    for n in range(n_days):
        start_date = datetime(time[0].year, time[0].month, time[0].day, 0, 0)+timedelta(days=n)
        end_date = start_date+timedelta(days=1)
        l_time = get_l_time_range(time, start_date, end_date)
        daily_time.append(start_date)
        daily_values.append(np.nanmean(values[l_time], axis=time_axis))

    return np.array(daily_time), np.array(daily_values)

def get_monthly_sums(time:np.ndarray, values:np.ndarray, time_axis=0) -> tuple:
    monthly_time = []
    monthly_values = []

    start_date = datetime(time[0].year, time[0].month, 1)
    end_date = datetime(time[-1].year, time[-1].month, 1)
    n_months = 0
    add_date = add_month_to_time(start_date, n_months)
    while add_date != end_date:
        n_months += 1
        add_date = add_month_to_time(start_date, n_months)

    for n in range(n_months+1):
        start_date = add_month_to_time(time[0], n)
        end_date = add_month_to_time(time[0], n+1)
        l_time = get_l_time_range(time, start_date, end_date)
        monthly_time.append(start_date+timedelta(seconds=(end_date-start_date).total_seconds()/2))
        monthly_values.append(np.nansum(values[l_time], axis=time_axis))

    return np.array(monthly_time), np.array(monthly_values)

def get_monthly_means(time:np.ndarray, values:np.ndarray, time_axis=0) -> tuple:
    monthly_time = []
    monthly_values = []

    start_date = datetime(time[0].year, time[0].month, 1)
    end_date = datetime(time[-1].year, time[-1].month, 1)
    n_months = 0
    add_date = add_month_to_time(start_date, n_months)
    while add_date != end_date:
        n_months += 1
        add_date = add_month_to_time(start_date, n_months)

    for n in range(n_months+1):
        start_date = add_month_to_time(time[0], n)
        end_date = add_month_to_time(time[0], n+1)
        l_time = get_l_time_range(time, start_date, end_date)
        monthly_time.append(start_date+timedelta(seconds=(end_date-start_date).total_seconds()/2))
        monthly_values.append(np.nanmean(values[l_time], axis=time_axis))

    return np.array(monthly_time), np.array(monthly_values)

def get_yearly_means(time:np.ndarray, values:np.ndarray, time_axis=0) -> tuple:
    yearly_time = []
    yearly_values = []
    
    n_years = time[-1].year - time[0].year + 1
    
    for n in range(n_years):
        start_date = datetime(time[0].year+n, 1, 1)
        end_date = datetime(start_date.year, 12, 31)
        l_time = get_l_time_range(time, start_date, end_date)
        yearly_time.append(datetime(start_date.year, 7, 2)) # middle of the year
        yearly_values.append(np.nanmean(values[l_time], axis=time_axis))
        
    return np.array(yearly_time), np.array(yearly_values)

def get_yearly_sums(time:np.ndarray, values:np.ndarray, time_axis=0) -> tuple:
    yearly_time = []
    yearly_values = []
    
    n_years = time[-1].year - time[0].year + 1
    
    for n in range(n_years):
        start_date = datetime(time[0].year+n, 1, 1)
        end_date = datetime(start_date.year, 12, 31)
        l_time = get_l_time_range(time, start_date, end_date)
        yearly_time.append(datetime(start_date.year, 7, 2)) # middle of the year
        yearly_values.append(np.nansum(values[l_time], axis=time_axis))
        
    return np.array(yearly_time), np.array(yearly_values)

def convert_time_to_datetime(time_org:np.ndarray, time_units:str) -> np.ndarray:
    time = []
    if 'since' in time_units:   
        i_start_time = time_units.index('since')+len('since')+1
    elif 'after' in time_units:
        i_start_time = time_units.index('after')+len('after')+1
    else:
        raise ValueError('Unknown time units: "since" or "after" not found in units.')
    if any(re.findall(f'\dT\d', time_units)): # YYYY-mm-ddTHH:MM format used by Parcels
        i_end_time = i_start_time+len('YYYY-mm-ddTHH:MM')
        base_time = datetime.strptime(time_units[i_start_time:i_end_time],'%Y-%m-%dT%H:%M')
    else: # YYYY-mm-dd format used by multiple numerical models
        i_end_time = i_start_time+len('YYYY-mm-dd')
        base_time = datetime.strptime(time_units[i_start_time:i_end_time],'%Y-%m-%d')
    if time_units.startswith('seconds'):
        for t in time_org:
            if not np.isnan(t):
                time.append(base_time+timedelta(seconds=t))
            else:
                time.append(np.nan)
        return np.array(time)
    elif time_units.startswith('hours'):
        for t in time_org:
            if not np.isnan(t):
                time.append(base_time+timedelta(hours=t))
            else:
                time.append(np.nan)
        return np.array(time)
    elif time_units.startswith('days'):
        for t in time_org:
            if not np.isnan(t):
                time.append(base_time+timedelta(days=t))
            else:
                time.append(np.nan)
        return np.array(time)
    else:
        raise ValueError('Unknown time units for time conversion to datetime.')

def convert_datetime_to_time(time_org:np.ndarray, time_units='seconds',
                             time_origin=datetime(1995,1,1,12,0)) -> np.ndarray:
    time = []
    if time_units == 'seconds':
        conversion = 1
    elif time_units == 'hours':
        conversion = 60*60
    elif time_units == 'days':
        conversion = 24*60*60
    else:
        raise ValueError('Unknown time units requested for time conversion from datetime.')
    for t in time_org:        
        time.append((t-time_origin).total_seconds()/conversion)
    return np.array(time), f'{time_units} since {time_origin.strftime("%Y-%m-%d")}'
