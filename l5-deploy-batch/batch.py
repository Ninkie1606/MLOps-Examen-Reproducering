import mlflow
from mlflow import MlflowClient
import pandas as pd
from prefect import flow, task
import os
import requests

client = MlflowClient("http://experiment-tracking:5000")
mlflow.set_tracking_uri("http://experiment-tracking:5000")
model_name = "rf-best-model"


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    return df


def prep_dicts(df):
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    return dicts


def get_latest_version(model_name):

    response = requests.post(
        "http://experiment-tracking:5000/api/2.0/mlflow/registered-models/get-latest-versions",
        json={"name": model_name, "stages": ["None"]},
    )

    latest_versions = response.json().get("model_versions", [])
    latest_version = latest_versions[-1]["version"]

    return latest_version


def load_model():
    print("...loading model")
    latest_version = get_latest_version(model_name)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

    return model


def load_data(year, month):
    print("...reading dataframe")
    df = read_dataframe(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet"
    )

    return df


@flow
def run_batch(year, month):
    model = load_model()
    df = load_data(year, month)

    print("...prepping data")
    dicts = prep_dicts(df)

    print("...calculating prediction")
    y_pred = model.predict(dicts)

    print("...creating new dataframe")
    df_result = pd.DataFrame()

    print(model.metadata.run_id)

    df_result["lpep_dropoff_datetime"] = df.lpep_dropoff_datetime
    df_result["PULocationID"] = df.PULocationID
    df_result["DOLocationID"] = df.DOLocationID
    df_result["duration"] = df.duration
    df_result["duration_pred"] = y_pred
    df_result["duration_delta"] = df_result.duration - df_result.duration_pred
    df_result["model_id"] = model.metadata.run_id

    print("...saving dataframe")

    path = f"/batch-data/report/green/{year:04d}/{month:02d}"
    print(f"...saving dataframe to: {path}")
    os.makedirs(path, exist_ok=True)
    df_result.to_parquet(f"{path}/{model.metadata.run_id}.parquet")


if __name__ == "__main__":
    run_batch.serve(
        name="batch-april",
        parameters={"year": 2021, "month": 4},
        work_pool_name="batch",
    )
