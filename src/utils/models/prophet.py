import pandas as pd
from prophet import Prophet as fb_Prophet

from .model import Model


class Prophet(Model):
    def __init__(self):
        self.model = None

    def fit_building_data(self,
                          train_data: pd.DataFrame,
                          train_data_covars: pd.DataFrame,
                          validation_data: pd.DataFrame,
                          validation_data_covars: pd.DataFrame,
                          hyperparams: dict,
                          early_stopping=True,
                          verbose=False) -> None:
        changepoint_prior_scale = hyperparams.get('changepoint_prior_scale', 0.05)
        weekly_seasonality = hyperparams.get('weekly_seasonality', 3)
        daily_seasonality = hyperparams.get('daily_seasonality', 4)

        self.model = fb_Prophet(changepoint_prior_scale=changepoint_prior_scale, weekly_seasonality=weekly_seasonality,
                                daily_seasonality=daily_seasonality)

        # merge covariates and add as regressors
        train_data_prophet = pd.merge(train_data, train_data_covars, on='ds')
        prophet_regressors = train_data_covars.columns.difference(['ds']).values
        for regressor in prophet_regressors:
            self.model.add_regressor(regressor)
        self.model.fit(train_data_prophet)

    def predict_day_for_building(self,
                                 input_data: pd.DataFrame,
                                 input_data_covars: pd.DataFrame,
                                 number_of_prediction_time_steps: int,
                                 prediction_start,
                                 prediction_day: int) -> pd.DataFrame:
        # create future dataset for prediction and pass to predict()
        # one column for ds, covering the prediction time span
        future = input_data_covars.loc[
            pd.to_datetime(input_data_covars.ds) >= (prediction_start + pd.Timedelta(days=prediction_day))
            ]
        prediction = self.model.predict(future).rename(columns={'yhat': 'y'})
        return prediction
