import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR as sklearn_SVR

from .model import Model


class SVR(Model):
    def __init__(self):
        self.model = None
        self.scaler_target = MinMaxScaler()
        self.scaler_covars = MinMaxScaler()

    def fit_building_data(self,
                          train_data: pd.DataFrame,
                          train_data_covars: pd.DataFrame,
                          validation_data: pd.DataFrame,
                          validation_data_covars: pd.DataFrame,
                          hyperparams: dict,
                          early_stopping=True,
                          verbose=False) -> None:
        regularization_param = hyperparams.get('regularization_param', 1)
        epsilon = hyperparams.get('epsilon', 0.1)
        input_length = 24

        y_train = train_data['y'].iloc[input_length:].values

        x_cols = []
        for i in range(input_length):
            x_cols.append(train_data['y'].shift(i + 1).iloc[input_length:])
        train_data_covars = train_data_covars.loc[
                            pd.to_datetime(train_data_covars.ds) <= max(pd.to_datetime(train_data.ds)), :
                            ]
        for i in range(1, len(train_data_covars.columns)):
            x_cols.append(train_data_covars.iloc[input_length:, i])
        x_train = pd.concat(x_cols, axis=1).to_numpy()

        y_train = y_train.reshape(-1, 1)
        self.scaler_target.fit(y_train)
        self.scaler_covars.fit(x_train)

        y_train = self.scaler_target.transform(y_train)
        y_train = y_train.reshape(1, -1)[0]
        x_train = self.scaler_covars.transform(x_train)

        self.model = sklearn_SVR(C=regularization_param, epsilon=epsilon)
        self.model.fit(x_train, y_train)

    def predict_day_for_building(self,
                                 input_data: pd.DataFrame,
                                 input_data_covars: pd.DataFrame,
                                 number_of_prediction_time_steps: int,
                                 prediction_start,
                                 prediction_day: int) -> pd.DataFrame:
        # predict time_steps values, starting with first time step after input_data
        input_data_covars = input_data_covars.loc[
                            pd.to_datetime(input_data_covars.ds) > max(pd.to_datetime(input_data.ds)), :]

        y = np.empty(number_of_prediction_time_steps)
        y[:] = np.nan
        for step in range(number_of_prediction_time_steps):
            # predict
            # use predicted value for subsequent predictions

            # get all the regressors for this prediction
            x = np.array([np.concatenate((
                input_data['y'].iloc[step:].values,  # observed output values (previous day)
                y[:step],  # predicted output values (earlier the same day)
                input_data_covars.iloc[step, 1:].values.astype(np.float64)  # covariates at the current time step
            ))])

            x_scaled = self.scaler_covars.transform(x)
            y[step] = self.scaler_target.inverse_transform(self.model.predict(x_scaled).reshape(1, -1))

        prediction = pd.DataFrame({'y': y})
        return prediction
