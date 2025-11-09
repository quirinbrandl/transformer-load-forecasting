import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from .predict_utils import predict_in_all_buildings


def objective_factory(data, sites_covariates, test_start, validation_length, prediction_length, daily_retrain, train_length,
                      model_type, model_hyperparam_candidates, hyperparams_fixed):
    def objective(trial):
        # hyperparameter search space
        if model_type == 'tft':
            hyperparams = {
                'hidden_size': trial.suggest_categorical('hidden_size', model_hyperparam_candidates['hidden_size']),
                'num_attention_heads': trial.suggest_categorical('num_attention_heads', model_hyperparam_candidates['num_attention_heads']),
                'lstm_layers': trial.suggest_categorical('lstm_layers', model_hyperparam_candidates['lstm_layers']),
                'full_attention': trial.suggest_categorical('full_attention', model_hyperparam_candidates['lstm_layers']),
                'dropout': trial.suggest_categorical('dropout', model_hyperparam_candidates['dropout']),
                'early_stopping_min_delta': hyperparams_fixed[model_type][train_length]['early_stopping_min_delta'],
                'early_stopping_patience': hyperparams_fixed[model_type][train_length]['early_stopping_patience']
            }
        elif model_type == 'transformer':
            hyperparams = {
                'd_model': trial.suggest_categorical('d_model', model_hyperparam_candidates['d_model']),
                'nhead': trial.suggest_categorical('nhead', model_hyperparam_candidates['nhead']),
                'num_encoder_layers': trial.suggest_categorical('num_encoder_layers',
                                                                model_hyperparam_candidates['num_encoder_layers']),
                'num_decoder_layers': trial.suggest_categorical('num_decoder_layers',
                                                                model_hyperparam_candidates['num_decoder_layers']),
                'dim_feedforward': trial.suggest_categorical('dim_feedforward',
                                                             model_hyperparam_candidates['dim_feedforward']),
                'dropout': trial.suggest_categorical('dropout', model_hyperparam_candidates['dropout']),
                'early_stopping_min_delta': hyperparams_fixed[model_type][train_length]['early_stopping_min_delta'],
                'early_stopping_patience': hyperparams_fixed[model_type][train_length]['early_stopping_patience']
            }
        elif model_type == 'lstm':
            hyperparams = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', model_hyperparam_candidates['hidden_dim']),
                'n_rnn_layer': trial.suggest_categorical('n_rnn_layer', model_hyperparam_candidates['n_rnn_layer']),
                'dropout': trial.suggest_categorical('dropout', model_hyperparam_candidates['dropout']),
                'early_stopping_min_delta': hyperparams_fixed[model_type][train_length]['early_stopping_min_delta'],
                'early_stopping_patience': hyperparams_fixed[model_type][train_length]['early_stopping_patience']
            }
        elif model_type == 'prophet':
            hyperparams = {
                'changepoint_prior_scale': trial.suggest_categorical('changepoint_prior_scale',
                                                                     model_hyperparam_candidates[
                                                                         'changepoint_prior_scale']),
                'weekly_seasonality': trial.suggest_categorical('weekly_seasonality',
                                                                model_hyperparam_candidates['weekly_seasonality']),
                'daily_seasonality': trial.suggest_categorical('daily_seasonality',
                                                               model_hyperparam_candidates['daily_seasonality'])
            }
        elif model_type == 'svr':
            hyperparams = {
                'regularization_param': trial.suggest_categorical('regularization_param',
                                                                  model_hyperparam_candidates['regularization_param']),
                'epsilon': trial.suggest_categorical('epsilon', model_hyperparam_candidates['epsilon'])
            }
        elif model_type == 'rf':
            hyperparams = {
                'n_estimators': trial.suggest_categorical('n_estimators', model_hyperparam_candidates['n_estimators'])
            }
        elif model_type == 'arima':
            hyperparams = {
                'p': trial.suggest_categorical('p', model_hyperparam_candidates['p']),
                'd': trial.suggest_categorical('d', model_hyperparam_candidates['d']),
                'q': trial.suggest_categorical('q', model_hyperparam_candidates['q']),
            }
        else:
            raise NotImplementedError(f'Unknown model type is provided: {model_type}')

        predications = predict_in_all_buildings(data=data, sites_covariates=sites_covariates,
                                                test_start=test_start, training_length=train_length,
                                                validation_length=validation_length,
                                                model_type=model_type, prediction_length=prediction_length,
                                                daily_retrain=daily_retrain,
                                                hyperparams=hyperparams)

        return calculate_metrics(predications, data)

    return objective


def calculate_metrics(predictions, data, error_metric='smape'):
    evaluation_data = pd.merge(predictions, data.rename(columns={'timestamp': 'ds'}, copy=True), on='ds',
                               suffixes=('_yhat', '_y'))
    buildings_metrics = []
    for building in range(1, predictions.shape[1] - 2):  # '-2' because last columns are model and training_length
        building_name = predictions.columns[building]
        if error_metric == 'mae':
            buildings_metrics.append(
                mean_absolute_error(evaluation_data[building_name + '_y'], evaluation_data[building_name + '_yhat']))
        elif error_metric == 'mse':
            buildings_metrics.append(
                mean_squared_error(evaluation_data[building_name + '_y'], evaluation_data[building_name + '_yhat']))
        elif error_metric == 'mape':
            buildings_metrics.append(mean_absolute_percentage_error(evaluation_data[building_name + '_y'],
                                                                    evaluation_data[building_name + '_yhat']))
        elif error_metric == 'smape':
            y = evaluation_data[building_name + '_y']
            yhat = evaluation_data[building_name + '_yhat']
            n = len(y) - (y.isna() | yhat.isna()).sum()
            if n == 0:
                buildings_metrics.append(0)
            else:
                num = (y - yhat).abs()
                den = y.abs() + yhat.abs()
                smape = num / den
                smape[den == 0] = 0
                smape = 200 * smape.sum() / n
                buildings_metrics.append(smape)
        elif error_metric == 'r2':
            buildings_metrics.append(
                r2_score(evaluation_data[building_name + '_y'], evaluation_data[building_name + '_yhat']))
        else:
            raise NotImplementedError(f'unknown error metric provided: {error_metric}')
    return np.mean(buildings_metrics)


def print_callback_factory(log_file):
    def print_callback(study, trial):
        print(f'Trial {trial.number}, current value: {trial.value}, current parameters: {trial.params}')
        log_file.write(f'Trial {trial.number}, current value: {trial.value}, current parameters: {trial.params}\n')
        print(f'Best value: {study.best_value}, Best parameters: {study.best_trial.params}')
        log_file.write(f'Best value: {study.best_value}, Best parameters: {study.best_trial.params}\n')
    return print_callback
