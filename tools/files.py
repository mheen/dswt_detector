from datetime import datetime, timedelta
import json
import os

def get_dir_from_json(dirname:str, json_file='input/dirs.json') -> str:
    with open(json_file, 'r') as f:
        all_dirs = json.load(f)
    return all_dirs[dirname]

def get_files_in_dir(input_dir:str, file_ext:str, return_full_path=True) -> list:
    files = []
    for filename in os.listdir(input_dir):
        if filename.endswith(f'.{file_ext}'):
            if return_full_path is True:
                files.append(f'{input_dir}{filename}')
            else:
                files.append(filename)
    return files

def get_daily_files_in_time_range(input_dir:str,
                                  start_date:datetime,
                                  end_date:datetime,
                                  file_ext:str,
                                  timeformat='%Y%m%d'):
    all_files = get_files_in_dir(input_dir, file_ext)
    ndays = (end_date-start_date).days+1
    files = []
    for n in range(ndays):
        date = start_date+timedelta(days=n)
        for file in all_files:
            if date.strftime(timeformat) in file:
                files.append(file)
    return files

def create_dir_if_does_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
