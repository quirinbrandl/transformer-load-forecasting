import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from icecream import ic

from utils.predict_utils import predict_in_all_buildings

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
default_out_dir = f"output/predictions/{timestamp}"

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config/config.yml')
parser.add_argument('--out_dir', default=default_out_dir)
args = parser.parse_args()
with open(args.config_file, 'r') as cfg_yaml:
    cfg = yaml.safe_load(cfg_yaml)

model_types: list[str] = cfg['model_types']
train_lengths: list[int] = cfg['train_lengths']
prediction_start_dates: list[str] = cfg['prediction_start_dates']
hyperparams: dict = cfg['hyperparams']
validation_length: int = cfg['val_length']
prediction_length: int = cfg['pred_period']  # number of days for which predictions should be calculated
daily_retrain: bool = cfg['daily_retrain']  # train a model for each day of predictions?
path_electricity_data_processed: str = cfg['path_electricity_data_processed']
path_covariates_prefix: str = cfg['path_covariates_prefix']
test_run: bool = cfg['test_run']

path_covariates_dir = Path(path_covariates_prefix)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(path_electricity_data_processed)
sites = set([building.split('_', 1)[0] for building in data.columns[1:]])
sites_covariates = {}
for site in sites:
    sites_covariates[site] = pd.read_csv(path_covariates_dir / f'{site}.csv')

if test_run:
    train_lengths = train_lengths[0:2]
    data = data.iloc[:, 0:3]
    prediction_start_dates = prediction_start_dates[0:1]

time_dicts = {'prediction_start_date': [],
              'model': [],
              'training_length': [],
              'process_time_ns': [],
              'thread_time_ns': [],
              'wall_time_ns': []}
start_process_time = time.process_time_ns()
start_thread_time = time.thread_time_ns()
start_wall_time = time.time_ns()

all_predictions = []
for model_type in model_types:
    ic(model_type)
    if model_type in hyperparams:
        hyperparams_model = hyperparams[model_type]
    else:
        ic(f'skipping {model_type}, no hyperparameters are provided in the config file')
        continue

    for prediction_start_date in prediction_start_dates:
        ic(prediction_start_date)
        # test_start: first time step for which predictions are calculated, which are then used to calculate evaluation metrics
        test_start = pd.to_datetime(prediction_start_date, format='%Y-%m-%d %H:%M:%S')
        # iterate over lengths of training data
        # at least one week (because of seasonalities...), at most 4 weeks
        for training_length in train_lengths:
            ic(training_length)

            prediction = predict_in_all_buildings(data=data, sites_covariates=sites_covariates,
                                                  test_start=test_start, training_length=training_length,
                                                  validation_length=validation_length,
                                                  model_type=model_type, prediction_length=prediction_length,
                                                  daily_retrain=daily_retrain,
                                                  hyperparams=hyperparams_model[training_length])
            all_predictions.append(prediction)
            finish_process_time = time.process_time_ns()
            finish_thread_time = time.thread_time_ns()
            finish_wall_time = time.time_ns()
            time_dicts['process_time_ns'].append(finish_process_time - start_process_time)
            time_dicts['thread_time_ns'].append(finish_thread_time - start_thread_time)
            time_dicts['wall_time_ns'].append(finish_wall_time - start_wall_time)
            time_dicts['prediction_start_date'].append(prediction_start_date)
            time_dicts['training_length'].append(training_length)
            time_dicts['model'].append(model_type)
            start_process_time = finish_process_time
            start_thread_time = finish_thread_time
            start_wall_time = finish_wall_time

all_predictions = pd.concat(all_predictions, ignore_index=True)
all_predictions.columns = data.columns.append(pd.Index(['model', 'training_length']))

all_predictions.to_csv(out_dir / 'predictions.csv', index=False)
pd.DataFrame.from_dict(time_dicts).to_csv(out_dir / 'times.csv', index=False)