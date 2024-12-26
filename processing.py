from tools import log
import pandas as pd
import numpy as np

# WRONG: every transect needs a daily value (zero if no DSWT) for mean f_dswt to work correctly
# so rather than removing entries, need to set them to zero!!!

def remove_positive_transport(df:pd.DataFrame) -> pd.DataFrame:
    '''Function to remove any positive (towards the coast) cross-shelf transport,
    as this is not associated with DSWT. A similar approach has also been taken
    by others (e.g. Luneva et al. 2020).'''
    i_to_replace = df[df['transport_dswt'] < 0.0].index
    df.loc[i_to_replace, 'f_dswt'] = 0.0
    df.loc[i_to_replace, ['vel_dswt', 'transport_dswt', 'ds', 'dz_dswt', 'lon_transport', 'lat_transport', 'depth_transport']] = np.nan

    return df

def remove_transport_starting_too_deep(df:pd.DataFrame, max_starting_depth=45.0) -> pd.DataFrame:
    '''Function to remove DSWT associated cross-shelf transport if it
    starts too deep (default 45 m depth or more).
    Note that the default is 45 m because the Wadjemup added transects
    start at 40 m, so to keep any information from these transects this
    was a necessary minimum depth. However, ideally the minimum depth
    should probably be shallower. This should be changed if I find a better
    way of dealing with transects around islands (see also the function below).
    This can perhaps also be included in the detection algorithm, rather than
    as a processing step in the future.
    
    Input:
    - df: pandas dataframe containing DSWT output (read from csv output file)
    - max_starting_depth: float with depth, if transport starts deeper than this it is removed
      default 45 m
    '''
    
    df_depth = df.groupby(['time', 'transect']).agg(depth=('depth_transport', 'min'))
    i_to_replace = df_depth[df_depth['depth'] > max_starting_depth].index
    
    df_mi = df.set_index(['time', 'transect'])
    
    df_mi.loc[i_to_replace, 'f_dswt'] = 0.0
    df_mi.loc[i_to_replace, ['vel_dswt', 'transport_dswt', 'ds', 'dz_dswt', 'lon_transport', 'lat_transport', 'depth_transport']] = np.nan
    
    df = df_mi.reset_index()
    
    return df

def remove_faulty_transport_around_islands(df:pd.DataFrame, island_transects:list) -> pd.DataFrame:
    '''Function removes transport when DSWT has been detected in island transects but not anywhere else.
    This is done because island transects can start deeper than transects along the coast,
    and this may result in drho/dx < 0 being falsely satisfied and DSWT being detected.
    If DSWT was not detected in any other transects, then we assume these detections were faulty
    and remove them. Ideally, this still needs a better way of handling with these situations.
    Potentially the solution that allows transects to be added in starting at a different depth
    is not the right solution after all. However, at least on the Wadjemup continental shelf,
    the transects of Wadjemup are important and removing them entirely is also not a great solution...
    
    Input:
    - df: pandas dataframe containing DSWT output (read from csv output file)
    - island_transects: list of strings containing the island transect names'''
    
    if island_transects is None:
        log.info('No island transects given, skipping island processing step.')
        return df
    
    l_island = df['transect'].isin(island_transects)
    
    df_temp = df.copy()
    df_temp['island'] = ''
    df_temp['island'][l_island] = 'Island'
    df_temp['island'][~l_island] = 'Not Island'
    df_rott_gpd = df_temp.groupby(['time', 'island']).agg(t=('transport_dswt', 'mean')).reset_index()
    df_rott_gpd_pivot = df_rott_gpd.pivot(columns='island', values='t', index='time')
    
    # if island transects have transport when no other transects do: remove them
    l_transport_island_nowhere_else = np.logical_and(np.isnan(df_rott_gpd_pivot['Not Island']), ~np.isnan(df_rott_gpd_pivot['Island']))
    times_to_replace = df_rott_gpd_pivot[l_transport_island_nowhere_else].index
    
    # if island transects have > 5 times transport than other transects: remove the island transects
    df_both = df_rott_gpd_pivot[np.logical_and(~np.isnan(df_rott_gpd_pivot['Not Island']), ~np.isnan(df_rott_gpd_pivot['Island']))]
    df_both['frac'] = df_both['Island'].values/df_both['Not Island'].values
    times_to_replace.append(df_both[df_both['frac'] > 5].index)
    
    i_to_replace = df[np.logical_and(df['time'].isin(times_to_replace), df['transect'].isin(island_transects))].index
    df.loc[i_to_replace, 'f_dswt'] = 0.0
    df.loc[i_to_replace, ['vel_dswt', 'transport_dswt', 'ds', 'dz_dswt', 'lon_transport', 'lat_transport', 'depth_transport']] = np.nan
    
    return df

def process_dswt_output(df:pd.DataFrame, output_path:str, island_transects=None):
    
    df = remove_positive_transport(df)
    df = remove_transport_starting_too_deep(df)
    df = remove_faulty_transport_around_islands(df, island_transects=island_transects)
    
    log.info(f'Writing processed output to file: {output_path}')
    df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    input_path = 'output/test_114-116E_33-31S/dswt_2017_unprocessed.csv'
    df = pd.read_csv(input_path)
    
    island_transects = pd.read_csv('input/transects/test_transects_islands.csv')['added_transects'].values
    
    output_path = 'output/test_114-116E_33-31S/dswt_2017.csv'
    process_dswt_output(df, output_path, island_transects)