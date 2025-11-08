import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RandomForest

from .model import Model


class RF(Model):
    def __init__(self):
        self.model = None
        self.scaler_target = Scaler()
        self.scaler_covars = Scaler()

    def fit_building_data(self,
                          train_data: pd.DataFrame,
                          train_data_covars: pd.DataFrame,
                          validation_data: pd.DataFrame,
                          validation_data_covars: pd.DataFrame,
                          hyperparams: dict,
                          early_stopping=True,
                          verbose=False) -> None:
        lags = hyperparams.get('lags', 24)
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', None)
        multi_models = hyperparams.get('multi_models', True)

        # get scaled series
        train_series = TimeSeries.from_dataframe(
            train_data,
            'ds', 'y', fill_missing_dates=True, freq='60min'
        )
        self.scaler_target.fit(train_series)
        train_scaled = self.scaler_target.transform(train_series)
        train_covars_series = TimeSeries.from_dataframe(
            train_data_covars,
            'ds', None, fill_missing_dates=True, freq='60min'
        )
        self.scaler_covars.fit(train_covars_series)
        train_covars_scaled = self.scaler_covars.transform(train_covars_series)

        self.model = RandomForest(lags=lags, lags_future_covariates=[0], n_estimators=n_estimators, max_depth=max_depth,
                                  multi_models=multi_models)

        self.model.fit(series=train_scaled, future_covariates=train_covars_scaled)

    def predict_day_for_building(self,
                                 input_data: pd.DataFrame,
                                 input_data_covars: pd.DataFrame,
                                 number_of_prediction_time_steps: int,
                                 prediction_start,
                                 prediction_day: int) -> pd.DataFrame:
        input_covars_series = TimeSeries.from_dataframe(
            input_data_covars,
            'ds', None, fill_missing_dates=True, freq='60min'
        )
        input_covars_scaled = self.scaler_covars.transform(input_covars_series)

        input_series = TimeSeries.from_dataframe(
            input_data,
            'ds', 'y', fill_missing_dates=True, freq='60min'
        )
        input_scaled = self.scaler_target.transform(input_series)

        prediction_scaled = self.model.predict(number_of_prediction_time_steps, input_scaled,
                                               future_covariates=input_covars_scaled)
        prediction = self.scaler_target.inverse_transform(prediction_scaled)
        prediction = prediction.to_dataframe().reset_index()
        return prediction
