import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from preprocess import dump_pickle, preprocess, read_dataframe
from train import load_pickle


def make_taxi_df(durations_minutes):
    now = pd.Timestamp("2021-01-01")
    return pd.DataFrame({
        "lpep_pickup_datetime": [now] * len(durations_minutes),
        "lpep_dropoff_datetime": [
            now + pd.Timedelta(minutes=d) for d in durations_minutes
        ],
        "PULocationID": [43] * len(durations_minutes),
        "DOLocationID": [215] * len(durations_minutes),
        "trip_distance": [float(i + 1) for i in range(len(durations_minutes))],
    })


def test_read_dataframe_filters_short_and_long_trips(tmp_path):
    df = make_taxi_df([0.5, 1.0, 30.0, 60.0, 61.0])
    path = tmp_path / "test.parquet"
    df.to_parquet(str(path))

    result = read_dataframe(str(path))

    assert len(result) == 3
    assert result["duration"].between(1, 60).all()


def test_read_dataframe_casts_location_ids_to_str(tmp_path):
    df = make_taxi_df([10.0])
    path = tmp_path / "test.parquet"
    df.to_parquet(str(path))

    result = read_dataframe(str(path))

    assert result["PULocationID"].dtype == object
    assert result["DOLocationID"].dtype == object


def test_preprocess_creates_pu_do_feature():
    df = pd.DataFrame({
        "PULocationID": ["43", "100"],
        "DOLocationID": ["215", "50"],
        "trip_distance": [3.0, 5.0],
    })
    dv = DictVectorizer()
    _, dv_fitted = preprocess(df, dv, fit_dv=True)

    assert "PU_DO=43_215" in dv_fitted.feature_names_
    assert "PU_DO=100_50" in dv_fitted.feature_names_


def test_preprocess_train_val_shapes_match():
    df_train = pd.DataFrame({
        "PULocationID": ["43", "100"],
        "DOLocationID": ["215", "50"],
        "trip_distance": [3.0, 5.0],
    })
    df_val = pd.DataFrame({
        "PULocationID": ["43"],
        "DOLocationID": ["215"],
        "trip_distance": [4.0],
    })
    dv = DictVectorizer()
    X_train, dv_fitted = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv_fitted, fit_dv=False)

    assert X_train.shape[1] == X_val.shape[1]


def test_pickle_roundtrip(tmp_path):
    data = {"key": [1, 2, 3], "label": "test"}
    path = str(tmp_path / "data.pkl")

    dump_pickle(data, path)
    loaded = load_pickle(path)

    assert loaded == data
