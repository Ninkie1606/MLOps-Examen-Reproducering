import os
import pickle
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error



def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_optimization(data_path: str, num_trials: int):

    mlflow.set_experiment("Random-Forest-HPO")
    mlflow.sklearn.autolog(disable=True)

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, 1),
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run():
            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)

            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    print("...optimizing params")
    run_optimization("./models/", 5)
