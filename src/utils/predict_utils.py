import pandas as pd

from .models.arima import ARIMA
from .models.lstm import LSTM
from .models.prophet import Prophet
from .models.rf import RF
from .models.svr import SVR
from .models.transformer import Transformer
from .models.tft import TFT


def predict_in_all_buildings(data,
                             sites_covariates,
                             test_start,
                             training_length,
                             validation_length,
                             model_type,
                             prediction_length,
                             daily_retrain,
                             hyperparams=None,
                             early_stopping=True,
                             verbose=False):
    if hyperparams is None:
        hyperparams = {}
    predictions = []
    sites = list(sites_covariates.keys())

    # number_of_models: number of models that should be trained to predict prediction_period days
    number_of_models = prediction_length
    if not daily_retrain:
        number_of_models = 1  # only one model for all predictions
    # prediction_length: number of days that are predicted with one model
    if daily_retrain:
        prediction_length = 1

    for model_index in range(number_of_models):
        # get training and validation data
        training_start = test_start - pd.Timedelta(days=training_length + validation_length) + pd.Timedelta(
            days=model_index * prediction_length) - pd.Timedelta(hours=1)
        prediction_start = test_start + pd.Timedelta(days=model_index)
        validation_start = prediction_start - pd.Timedelta(days=validation_length)
        prediction_end = prediction_start + pd.Timedelta(days=prediction_length)
        validation_data = data.loc[(pd.to_datetime(data.timestamp) < prediction_start) & (
                    pd.to_datetime(data.timestamp) >= validation_start)].rename(columns={'timestamp': 'ds'}).copy()
        train_data = data.loc[(pd.to_datetime(data.timestamp) < validation_start) & (
                    pd.to_datetime(data.timestamp) >= training_start)].rename(columns={'timestamp': 'ds'}).copy()
        test_data = data.loc[(pd.to_datetime(data.timestamp) >= prediction_start) & (
                    pd.to_datetime(data.timestamp) < prediction_end)].rename(columns={'timestamp': 'ds'}).copy()
        train_data_covars = {}
        validation_data_covars = {}
        for site in sites:
            # add one day at the end, so that the values can be shifted by 24 hours without losing the last day;
            # the shifted values are used as past covariates for transformer
            train_data_covars[site] = sites_covariates[site][(
                                                                     pd.to_datetime(
                                                                         sites_covariates[site].timestamp) < (
                                                                                 validation_start + pd.Timedelta(
                                                                             days=1))) & (pd.to_datetime(
                sites_covariates[site].timestamp) >= training_start
                                                                                          )].rename(
                columns={'timestamp': 'ds'}).copy()
            validation_data_covars[site] = sites_covariates[site][(
                                                                          pd.to_datetime(
                                                                              sites_covariates[site].timestamp) < (
                                                                                      prediction_start + pd.Timedelta(
                                                                                  days=1))) & (pd.to_datetime(
                sites_covariates[site].timestamp) >= validation_start
                                                                                               )].rename(
                columns={'timestamp': 'ds'}).copy()

        # list for results of the following loop: one column for each building (+ one time columns), one row for each time step day model_index
        building_predictions = []
        building_predictions.append(test_data['ds'].reset_index(drop=True).copy())

        # input_chunk_length: one day
        #   one week difficult: would require more than one week of training data and more than one week of validation data
        #   (both training and validation data must be larger than input chunk length)
        # output_chunk_length: one day (or shorter???)
        for building in train_data.columns[1:]:
            # get for this building, based on building site (based on building name)
            site_id = building.split('_', 1)[0]
            train_data_covars_site = train_data_covars[site_id].copy()
            validation_data_covars_site = validation_data_covars[site_id].copy()
            train_data_building = train_data.loc[:, ['ds', building]].rename(columns={building: 'y'}).copy()
            validation_data_building = validation_data.loc[:, ['ds', building]].rename(columns={building: 'y'}).copy()

            # initialize model
            if model_type == 'tft':
                model = TFT()
            elif model_type == 'transformer':
                model = Transformer()
            elif model_type == 'lstm':
                model = LSTM()
            elif model_type == 'prophet':
                model = Prophet()
            elif model_type == 'arima':
                model = ARIMA()
            elif model_type == 'svr':
                model = SVR()
            elif model_type == 'rf':
                model = RF()
            else:
                raise NotImplementedError(f'The unknown model type is provided: {model_type}!')

            model.fit_building_data(train_data=train_data_building,
                                    train_data_covars=train_data_covars_site,
                                    validation_data=validation_data_building,
                                    validation_data_covars=validation_data_covars_site,
                                    hyperparams=hyperparams,
                                    early_stopping=early_stopping,
                                    verbose=verbose)

            building_predictions_model = []
            for current_prediction_day in range(prediction_length):
                input_data_covars = sites_covariates[site_id].loc[
                    (pd.to_datetime(sites_covariates[site_id].timestamp) >= (
                                prediction_start - pd.Timedelta(days=1) + pd.Timedelta(days=current_prediction_day))) &
                    (pd.to_datetime(sites_covariates[site_id].timestamp) < (
                                prediction_start + pd.Timedelta(days=current_prediction_day + 1)))
                    ].rename(columns={'timestamp': 'ds'}).copy()

                # make sure the input series is of length input_chunk_length and ends right before first prediction!
                input_data = data.loc[
                    (pd.to_datetime(data.timestamp) >= (
                                prediction_start - pd.Timedelta(days=1) + pd.Timedelta(days=current_prediction_day))) &
                    (pd.to_datetime(data.timestamp) < (prediction_start + pd.Timedelta(days=current_prediction_day))), [
                        'timestamp', building]
                ].rename(columns={'timestamp': 'ds', building: 'y'})

                # for each model type, make sure past/future covariates set to None if not supported by the model
                predictions_day = model.predict_day_for_building(input_data=input_data,
                                                                 input_data_covars=input_data_covars,
                                                                 number_of_prediction_time_steps=24,
                                                                 prediction_start=prediction_start,
                                                                 prediction_day=current_prediction_day)
                building_predictions_model.append(predictions_day['y'])

            building_predictions.append(pd.concat(building_predictions_model, ignore_index=True))

        building_predictions = pd.concat(building_predictions, axis=1, ignore_index=True)
        building_predictions.columns = train_data.columns
        building_predictions['model'] = model_type

        predictions.append(building_predictions)

    predictions = pd.concat(predictions, ignore_index=True)
    predictions['training_length'] = training_length

    return predictions
