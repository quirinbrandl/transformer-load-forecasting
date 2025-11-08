import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import ARIMA as darts_ARIMA

from .model import Model


class ARIMA(Model):
    def __init__(self):
        self.model = None
        self.with_covars = False
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
        p = hyperparams.get('p', 0)
        d = hyperparams.get('d', 0)
        q = hyperparams.get('q', 0)
        self.with_covars = hyperparams.get('with_covars', True)
        season = hyperparams.get('season', 0)
        p_season = hyperparams.get('p_season', 0)
        d_season = hyperparams.get('d_season', 0)
        q_season = hyperparams.get('q_season', 0)

        # trend = 'n' means that no deterministic trend is considered
        # assumption that there is no trend is ok because prediction horizon is only 24 hours -> there should not be a noticeable trend over one day
        self.model = darts_ARIMA(p=p, d=d, q=q, seasonal_order=(p_season, d_season, q_season, season), trend='n')

        train_series = TimeSeries.from_dataframe(
            train_data,
            'ds', 'y', fill_missing_dates=True, freq='60min'
        )
        self.scaler_target.fit(train_series)
        train_scaled = self.scaler_target.transform(train_series)

        if self.with_covars:
            train_covars_series = TimeSeries.from_dataframe(
                train_data_covars,
                'ds', None, fill_missing_dates=True, freq='60min'
            )
            self.scaler_covars.fit(train_covars_series)
            train_covars_scaled = self.scaler_covars.transform(train_covars_series)
            self.model.fit(series=train_scaled, future_covariates=train_covars_scaled)
        else:
            self.model.fit(series=train_scaled)

    def predict_day_for_building(self,
                                 input_data: pd.DataFrame,
                                 input_data_covars: pd.DataFrame,
                                 number_of_prediction_time_steps: int,
                                 prediction_start,
                                 prediction_day: int) -> pd.DataFrame:
        input_series = TimeSeries.from_dataframe(
            input_data,
            'ds', 'y', fill_missing_dates=True, freq='60min'
        )
        input_scaled = self.scaler_target.transform(input_series)

        if self.with_covars:
            input_covars_series = TimeSeries.from_dataframe(
                input_data_covars,
                'ds', None, fill_missing_dates=True, freq='60min'
            )
            input_covars_scaled = self.scaler_covars.transform(input_covars_series)
            prediction_scaled = self.model.predict(number_of_prediction_time_steps, input_scaled,
                                               future_covariates=input_covars_scaled)
        else:
            prediction_scaled = self.model.predict(number_of_prediction_time_steps, input_scaled)

        prediction = self.scaler_target.inverse_transform(prediction_scaled)
        prediction = prediction.to_dataframe().reset_index()
        return prediction
