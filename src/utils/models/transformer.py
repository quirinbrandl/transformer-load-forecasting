import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .model import Model


class Transformer(Model):
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
        output_chunk_length = hyperparams.get('output_chunk_length', 24)
        d_model = hyperparams.get('d_model', 4)
        nhead = hyperparams.get('nhead', 2)
        num_encoder_layers = hyperparams.get('num_encoder_layers', 3)
        num_decoder_layers = hyperparams.get('num_decoder_layers', 2)
        dim_feedforward = hyperparams.get('dim_feedforward', 512)
        dropout = hyperparams.get('dropout', 0.15)
        batch_size = hyperparams.get('batch_size', 32)
        n_epochs = hyperparams.get('n_epochs', 200)
        early_stopping_patience = n_epochs
        early_stopping_min_delta = hyperparams.get('early_stopping_min_delta', 0.01)
        if early_stopping:
            early_stopping_patience = hyperparams.get('early_stopping_patience', 10)

        # add lag for past covariates to use covariates as past covariates (transformer only allows past covariates)
        for col in train_data_covars.columns[1:]:
            train_data_covars[col] = train_data_covars[col].shift(-24)
        train_data_covars = train_data_covars.dropna(subset=['weekend'])
        for col in validation_data_covars.columns[1:]:
            validation_data_covars[col] = validation_data_covars[col].shift(-24)
        validation_data_covars = validation_data_covars.dropna(subset=['weekend'])

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

        # get scaled series
        validation_series = TimeSeries.from_dataframe(
            validation_data,
            'ds', 'y', fill_missing_dates=True, freq='60min'
        )
        validation_scaled = self.scaler_target.transform(validation_series)
        train_scaled = self.scaler_target.transform(train_series)
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

        self.model = TransformerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            # the model returns this number of distinct values --> make sure that output_chunk_length values are predicted at a time (m.predict(output_chunk_length, ...))
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_name="transf",
            # nr_epochs_val_period=10, # Number of epochs to wait before evaluating the validation loss; if this is set, val_loss is not available for early stopping
            d_model=d_model,
            nhead=nhead,
            # larger value leads to more parameters -> takes longer to train, model gets more complex / more specific patterns can be detected, overfitting risk?
            # intuitively: each attention head learns one pattern (in time series: relationship of the predicted value with a value among the input values???)
            num_encoder_layers=num_encoder_layers,
            # encoder layers: iterations to refine the cross attention weights
            num_decoder_layers=num_decoder_layers,
            # decoder layers: iterations to refine the influence of cross attention on the prediction (based on self attention in the input series)
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",  # default
            # loss_fn=MSELoss(), # this is already the default loss...
            random_state=42,
            save_checkpoints=False,
            force_reset=True,  # replace all existing checkpoints with the same model name
            pl_trainer_kwargs={'callbacks': [early_stopper]}
        )

        self.model.fit(series=train_scaled, past_covariates=train_covars_scaled, val_series=validation_scaled,
                       val_past_covariates=validation_covars_scaled, verbose=verbose)

    def predict_day_for_building(self,
                                 input_data: pd.DataFrame,
                                 input_data_covars: pd.DataFrame,
                                 number_of_prediction_time_steps: int,
                                 prediction_start,
                                 prediction_day: int) -> pd.DataFrame:

        # shift for using covariates as past covariates (transformer only allows past covariates)
        for col in input_data_covars.columns[1:]:
            input_data_covars[col] = input_data_covars[col].astype(float).shift(-24)
        input_data_covars = input_data_covars.dropna(subset=['weekend'])
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
                                               past_covariates=input_covars_scaled)
        prediction = self.scaler_target.inverse_transform(prediction_scaled)
        prediction = prediction.to_dataframe().reset_index()
        return prediction
