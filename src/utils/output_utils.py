import os

import pandas as pd


def join_outputs(
    time_of_runs: list[str], predictions_dir: str, save=False
) -> pd.DataFrame:
    prediction_dfs = [
        pd.read_csv(os.path.join(predictions_dir, time_of_run, "predictions.csv"))
        for time_of_run in time_of_runs
    ]
    pred_joined_df = pd.concat(prediction_dfs)

    times_dfs = [
        pd.read_csv(os.path.join(predictions_dir, time_of_run, "times.csv"))
        for time_of_run in time_of_runs
    ]
    times_joined_df = pd.concat(times_dfs)

    if save:
        predictions_output_path = os.path.join(predictions_dir, "predictions.csv")
        pred_joined_df.to_csv(predictions_output_path)

        times_output_path = os.path.join(predictions_dir, "times.csv")
        times_joined_df.to_csv(times_output_path)

    return pred_joined_df, times_joined_df
