import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from icecream import ic

from utils.weather_data import get_weather_data

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config/config.yml')
args = parser.parse_args()
with open(args.config_file, 'r') as cfg_yaml:
    cfg = yaml.safe_load(cfg_yaml)

use_weather_data_from_dataset: bool = cfg['use_weather_metadata']
path_metadata: str = cfg['path_metadata']
path_electricity_data_processed: str = cfg['path_electricity_data_processed']
path_electricity_data_unprocessed: str = cfg['path_electricity_data_unprocessed']
path_covariates_prefix: str = cfg['path_covariates_prefix']
path_weather_data: str = cfg['path_weather_data']

path_covariates_dir = Path(path_covariates_prefix)
path_covariates_dir.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(path_metadata)
meter_data = pd.read_csv(path_electricity_data_unprocessed)
weather_data = pd.read_csv(path_weather_data)

## preprocessing data

# count missing points in each column
n_missing_points_column = meter_data.isnull().sum()

meter_data = meter_data.loc[:, n_missing_points_column == 0]


# drop all buildings with close-to-zero energy consumption (< 1 kWh)
def count_small_values(column):
    return np.sum(column < 1)


no_small_values_columns = ~pd.concat([pd.Series([False]), meter_data.iloc[:, 1:].apply(count_small_values, axis=0) > 0])
meter_data = meter_data.iloc[:, no_small_values_columns.tolist()]

# covariates: weekend indicator
data_covars = meter_data[['timestamp']].copy()
data_covars['weekend'] = pd.to_datetime(data_covars['timestamp']).dt.dayofweek >= 5
data_covars['weekend'] = data_covars['weekend'].astype(int)

# covariates: weather data
weather_start = datetime.datetime.strftime(pd.to_datetime(min(data_covars.timestamp)) - pd.Timedelta(days=1),
                                           format="%Y-%m-%d %H:%M:%S")
weather_end = max(data_covars.timestamp)

# get information about sites
metadata = metadata.loc[metadata.building_id.isin(meter_data.columns), ['building_id', 'site_id', 'lat', 'lng']]
sites_metadata = metadata[['site_id', 'lat', 'lng']].drop_duplicates()

# make sure that each site has exactly one lat/long combination
if sites_metadata['site_id'].nunique() != len(sites_metadata):
    ic(sites_metadata)
    ic(sites_metadata['site_id'].nunique())
    ic(len(sites_metadata))
    raise ValueError('Each site_id should map to exactly one (lat, lng) pair!')

# add weather data to covars
for index, row in sites_metadata.iterrows():
    if use_weather_data_from_dataset:
        weather_data_site = weather_data.loc[weather_data.site_id == row['site_id'], weather_data.columns != 'site_id']
        data_covars_site = pd.merge(data_covars, weather_data_site, how='left', on='timestamp')
    else:
        lat = row['lat']
        lng = row['lng']
        weather_data_site = get_weather_data(lat=lat, lng=lng, start_date=weather_start, end_date=weather_end)
        data_covars_site = pd.merge(data_covars, weather_data_site, how='left', on='timestamp').copy()
    # save data covars per site
    data_covars_site.to_csv(path_covariates_dir / f'{row['site_id']}.csv', index=False)

# save clean meter data
meter_data.to_csv(path_electricity_data_processed, index=False)
