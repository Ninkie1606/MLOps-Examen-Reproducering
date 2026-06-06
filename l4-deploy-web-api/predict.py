import mlflow
import requests
from flask import Flask, request, jsonify

app = Flask("duration-prediction")

mlflow.set_tracking_uri("http://experiment-tracking:5000")
# rf-best-model/versions/2
model = mlflow.pyfunc.load_model("models:/rf-best-model/latest")


def get_latest_version(model_name):

    response = requests.post(
        "http://experiment-tracking:5000/api/2.0/mlflow/registered-models/get-latest-versions",
        json={"name": model_name, "stages": ["None"]},
    )

    latest_versions = response.json().get("model_versions", [])
    latest_version = latest_versions[-1]["version"]

    print(latest_versions)

    return latest_version


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    get_latest_version("rf-best-model")

    ride = request.get_json()

    features = {}
    features["PU_DO"] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]

    predictions = model.predict([features])
    current_prediction = float(predictions[0])

    result = {"duration": current_prediction}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)


# {
#     "PULocationID": 10,
#     "DOLocationID": 50,
#     "trip_distance": 40
# }
