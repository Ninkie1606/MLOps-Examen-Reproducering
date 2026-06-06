import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from sklearn.pipeline import make_pipeline

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

HPO_EXPERIMENT_NAME = "random-forest-hpo"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs",
]

mlflow.set_tracking_uri("http://experiment-tracking:5000")


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    return dicts


def train_and_log_model(params):
    df_train = read_dataframe("data/green_tripdata_2021-01.parquet")
    df_val = read_dataframe("data/green_tripdata_2021-02.parquet")

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    dict_train = prepare_dictionaries(df_train)
    dict_val = prepare_dictionaries(df_val)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        pipeline = make_pipeline(DictVectorizer(), RandomForestRegressor(**params))
        pipeline.fit(dict_train, y_train)
        y_pred = pipeline.predict(dict_val)

        rmse = root_mean_squared_error(y_pred, y_val)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")


def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )
    for run in runs:
        train_and_log_model(params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse DESC"],
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="rf-best-model")


if __name__ == "__main__":
    print("...registering model")
    run_register_model("./data/", 5)
