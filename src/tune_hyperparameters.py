import argparse
import json
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
import yaml
from icecream import ic

from utils.hyperparameters_tuning_utils import objective_factory, print_callback_factory

RANDOM_STATE = 68794  # for sampling buildings

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
default_out_dir = f"../output/hyperparameter_tuning/{timestamp}"

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config.yml')
parser.add_argument('--out_dir', default=default_out_dir)
args = parser.parse_args()
with open(args.config_file, 'r') as cfg_yaml:
    cfg = yaml.safe_load(cfg_yaml)

hyperparam_tuning_start: str = cfg['hyperparam_tuning_start']
train_lengths: list[int] = cfg['train_lengths']
model_types: list[str] = cfg['model_types']
hyperparam_candidates: dict = cfg['hyperparam_candidates']
hyperparams_fixed: dict = cfg['hyperparams']
validation_length: int = cfg['val_length']
prediction_length: int = cfg['pred_period']
daily_retrain: bool = cfg['daily_retrain']
path_electricity_data_processed: str = cfg['path_electricity_data_processed']
path_metadata: str = cfg['path_metadata']
path_covariates_prefix: str = cfg['path_covariates_prefix']
test_run: bool = cfg['test_run']

path_covariates_dir = Path(path_covariates_prefix)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(path_electricity_data_processed)
metadata = pd.read_csv(path_metadata)
sites = set([building.split('_', 1)[0] for building in data.columns[1:]])
sites_covariates = {}
for site in sites:
    sites_covariates[site] = pd.read_csv(path_covariates_dir / f'{site}.csv')
test_start = pd.to_datetime(hyperparam_tuning_start, format='%Y-%m-%d %H:%M:%S')

# consider only metadata for buildings that are also in preprocessed data
metadata = metadata.loc[metadata.building_id.isin(data.columns), :]

# select subset of buildings
# random state in sample function is set to make results reproducible
n_buildings = 10
# quota sample based on primary use category; categories with share of <10% are aggregated (to category 'other'); at least 1 sample for each category
building_category_shares = metadata.groupby('primaryspaceusage')['primaryspaceusage'].count() / metadata[
    'primaryspaceusage'].count()  # share of each usage category
building_category_shares = pd.DataFrame({'usage_cat': building_category_shares.index,
                                         'share': building_category_shares.values})  # share of each usage category (dataframe)
# these shares slightly differ from those visualized on the project homepage; there, share among all data points is considered, here only among buildings with a non-missing use category
other_categories = list(building_category_shares.loc[building_category_shares.share < 0.1, :].usage_cat)
metadata.loc[metadata.primaryspaceusage.isin(other_categories), 'primaryspaceusage'] = 'Other'
building_category_shares.loc[building_category_shares.usage_cat.isin(other_categories), 'usage_cat'] = 'Other'
shares_agg = building_category_shares.groupby('usage_cat').sum()
building_category_shares = pd.DataFrame(
    {'usage_cat': shares_agg.iloc[:, 0].index, 'share': shares_agg.iloc[:, 0].values})
building_category_shares['n_samples'] = round(n_buildings * building_category_shares.share).astype(int)
building_category_shares.loc[building_category_shares.n_samples < 1, 'n_samples'] = 1
buildings_list = []
for i in range(len(building_category_shares)):
    buildings_in_category = metadata.loc[
        metadata.primaryspaceusage == building_category_shares.iloc[i].usage_cat, 'building_id']
    if len(buildings_in_category) < 1:
        continue
    if len(buildings_in_category) < building_category_shares.iloc[i].n_samples:
        # fewer buildings available than would be necessary according to shares -> select all buildings
        buildings_list.append(buildings_in_category.sample(n=len(buildings_in_category), random_state=RANDOM_STATE))
    else:
        buildings_list.append(
            buildings_in_category.sample(n=building_category_shares.iloc[i].n_samples, random_state=RANDOM_STATE))
selected_buildings = pd.concat(buildings_list, ignore_index=True)

if test_run:
    selected_buildings = selected_buildings.iloc[0:2]

selected_columns = ['timestamp']
selected_columns.extend(list(selected_buildings.values))
data = data.loc[:, selected_columns]

if test_run:
    train_lengths = [14, 28]
    prediction_length = 2

for model_type in model_types:
    # check if the dict contains an entry with model_type as key
    if model_type in hyperparam_candidates:
        model_hyperparam_candidates = hyperparam_candidates[model_type]
    else:
        ic(f'skipping {model_type}, no hyperparameter candidates are provided in the config file')
        continue

    log_file_path = out_dir / f'{model_type}_hyperparam_tuning_stats.log'
    with open(log_file_path, 'w') as log_file:
        ic('')
        ic('')
        ic('################################################################')
        ic('################################################################')
        ic(model_type)
        ic('################################################################')
        ic('################################################################')
        ic('')
        ic('')
        ic(model_hyperparam_candidates)

        max_tune_iters = model_hyperparam_candidates['max_tune_iters']
        ic(max_tune_iters)
        exhaustive_search = model_hyperparam_candidates['exhaustive_search']
        ic(exhaustive_search)

        if test_run:
            max_tune_iters = 2

        train_lengths_params = {}
        for train_length in train_lengths:
            ic('')
            ic('################################################################')
            ic(train_length)
            ic('################################################################')
            ic('')
            log_file.write(f'train_length: {train_length}\n')

            if exhaustive_search:
                hyperparams_for_search = {k: v for k, v in model_hyperparam_candidates.items() if
                                          k not in ['max_tune_iters', 'exhaustive_search']}
                required_iterations = 1
                for value in hyperparams_for_search.values():
                    required_iterations = required_iterations * len(value)
                if max_tune_iters < required_iterations and not test_run:
                    raise ValueError(
                        'Exhaustive search not possible; more combinations of hyperparameter values than n_trials')
                study = optuna.create_study(direction='minimize',
                                            sampler=optuna.samplers.GridSampler(hyperparams_for_search))
            else:
                study = optuna.create_study(direction='minimize')
            objective_function = objective_factory(data=data, sites_covariates=sites_covariates, test_start=test_start,
                                                   validation_length=validation_length,
                                                   prediction_length=prediction_length,
                                                   daily_retrain=daily_retrain, train_length=train_length,
                                                   model_type=model_type,
                                                   model_hyperparam_candidates=model_hyperparam_candidates,
                                                   hyperparams_fixed=hyperparams_fixed)
            study.optimize(objective_function, n_trials=max_tune_iters, callbacks=[print_callback_factory(log_file)])
            train_lengths_params[train_length] = study.best_params

    with open(out_dir / f'{model_type}_tuned_hyperparams.json', 'w') as f:
        json.dump(train_lengths_params, f)
