import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import MSELoss

from .model import Model


class TFT(Model):
    def __init__(self):
        self.model = None
        self.scaler_target = Scaler()
        self.scaler_covars = Scaler()

    def fit_building_data(
        self,
        train_data: pd.DataFrame,
        train_data_covars: pd.DataFrame,
        validation_data: pd.DataFrame,
        validation_data_covars: pd.DataFrame,
        hyperparams: dict,
        early_stopping=True,
        verbose=False,
    ) -> None:
        input_chunk_length = hyperparams.get("input_chunk_length", 24)
        output_chunk_length = hyperparams.get("output_chunk_length", 24)
        dropout = hyperparams.get("dropout", 0.15)
        hidden_size = hyperparams.get("hidden_size", 64)
        lstm_layers = hyperparams.get("lstm_layers", 1)
        batch_size = hyperparams.get("batch_size", 32)
        n_epochs = hyperparams.get("n_epochs", 200)
        early_stopping_patience = n_epochs
        early_stopping_min_delta = hyperparams.get("early_stopping_min_delta", 0.01)
        if early_stopping:
            early_stopping_patience = hyperparams.get("early_stopping_patience", 10)

        # fit transformers (determine parameters for normalizing the data)
        # based only on training data to avoid leakage (of information from other than the training data...)
        train_series = TimeSeries.from_dataframe(
            train_data, "ds", "y", fill_missing_dates=True, freq="60min"
        )
        self.scaler_target.fit(train_series)
        train_covars_series = TimeSeries.from_dataframe(
            train_data_covars, "ds", None, fill_missing_dates=True, freq="60min"
        )
        self.scaler_covars.fit(train_covars_series)

        # get scaled series
        validation_series = TimeSeries.from_dataframe(
            validation_data, "ds", "y", fill_missing_dates=True, freq="60min"
        )
        validation_scaled = self.scaler_target.transform(validation_series)
        train_scaled = self.scaler_target.transform(train_series)
        train_covars_scaled = self.scaler_covars.transform(train_covars_series)
        validation_covars_series = TimeSeries.from_dataframe(
            validation_data_covars, "ds", None, fill_missing_dates=True, freq="60min"
        )
        validation_covars_scaled = self.scaler_covars.transform(
            validation_covars_series
        )

        # early stopping configuration
        early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            # for stopping rule when training (number of epochs with loss decrease smaller than stopping_min_delta)
            min_delta=early_stopping_min_delta,
            # minimum delta based on building with minimum energy consumption: at least 1% of this building's average...
            mode="min",  # stop training when quantity monitored has stopped decreasing
            check_on_train_epoch_end=True,
        )

        self.model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_name="tft",
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            # TFT is probabilistic by default. To ensure comparability, make it deterministic
            likelihood=None,
            loss_fn=MSELoss(),
            use_static_covariates=False,
            random_state=42,
            save_checkpoints=False,
            force_reset=True,
            pl_trainer_kwargs={"callbacks": [early_stopper]},
        )

        self.model.fit(
            series=train_scaled,
            future_covariates=train_covars_scaled,
            val_series=validation_scaled,
            val_future_covariates=validation_covars_scaled,
            verbose=verbose,
        )

    def predict_day_for_building(
        self,
        input_data: pd.DataFrame,
        input_data_covars: pd.DataFrame,
        number_of_prediction_time_steps: int,
        prediction_start,
        prediction_day: int,
    ) -> pd.DataFrame:
        input_covars_series = TimeSeries.from_dataframe(
            input_data_covars, "ds", None, fill_missing_dates=True, freq="60min"
        )
        input_covars_scaled = self.scaler_covars.transform(input_covars_series)

        input_series = TimeSeries.from_dataframe(
            input_data, "ds", "y", fill_missing_dates=True, freq="60min"
        )
        input_scaled = self.scaler_target.transform(input_series)

        prediction_scaled = self.model.predict(
            number_of_prediction_time_steps,
            input_scaled,
            future_covariates=input_covars_scaled,
        )
        prediction = self.scaler_target.inverse_transform(prediction_scaled)
        prediction = prediction.to_dataframe().reset_index()
        return prediction
