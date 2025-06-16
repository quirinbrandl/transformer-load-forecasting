import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .model import Model


class LSTM(Model):
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
        input_chunk_length = hyperparams.get('input_chunk_length', 24)
        hidden_dim = hyperparams.get('hidden_dim', 25)
        n_rnn_layers = hyperparams.get('n_rnn_layers', 1)
        training_length = hyperparams.get('training_length', 25)
        dropout = hyperparams.get('dropout', 0.15)
        batch_size = hyperparams.get('batch_size', 32)
        n_epochs = hyperparams.get('n_epochs', 200)
        early_stopping_patience = n_epochs
        early_stopping_min_delta = hyperparams.get('early_stopping_min_delta', 0.01)
        if early_stopping:
            early_stopping_patience = hyperparams.get('early_stopping_patience', 10)

        # fit transformers (determine parameters for normalizing the data)
        # based only on training data to avoid leakage (of information from other than the training data...)
        train_series = TimeSeries.from_dataframe(
            train_data,
            'ds', 'y', fill_missing_dates=True, freq='60min'
        )
        self.scaler_target.fit(train_series)
        train_covars_series = TimeSeries.from_dataframe(
            train_data_covars,
            'ds', None, fill_missing_dates=True, freq='60min'
        )
        self.scaler_covars.fit(train_covars_series)

        # get scaled series (make sure that scaler is not fit again! scaler only used for transforming/normalizing data)
        train_scaled = self.scaler_target.transform(train_series)
        validation_series = TimeSeries.from_dataframe(
            validation_data,
            'ds', 'y', fill_missing_dates=True, freq='60min'
        )
        validation_scaled = self.scaler_target.transform(validation_series)
        train_covars_scaled = self.scaler_covars.transform(train_covars_series)
        validation_covars_series = TimeSeries.from_dataframe(
            validation_data_covars,
            'ds', None, fill_missing_dates=True, freq='60min'
        )
        validation_covars_scaled = self.scaler_covars.transform(validation_covars_series)

        # early stopping configuration
        early_stopper = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            # for stopping rule when training (number of epochs with loss decrease smaller than stopping_min_delta)
            min_delta=early_stopping_min_delta,
            # minimum delta based on building with minimum energy consumption: at least 1% of this building's average...
            mode='min',  # stop training when quantity monitored has stopped decreasing
            check_on_train_epoch_end=True
        )

        self.model = RNNModel(
            input_chunk_length=input_chunk_length,
            model='LSTM',
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            training_length=training_length,
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer_kwargs={"lr": 1e-3},
            model_name="lstm",
            random_state=42,
            log_tensorboard=False,
            force_reset=True,
            save_checkpoints=True,
            pl_trainer_kwargs={'callbacks': [early_stopper]}
        )

        self.model.fit(series=train_scaled, future_covariates=train_covars_scaled, val_series=validation_scaled,
                       val_future_covariates=validation_covars_scaled, verbose=verbose)

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
        prediction = prediction.pd_dataframe().reset_index()
        return prediction
