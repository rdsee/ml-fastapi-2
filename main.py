import datetime
from fastapi import FastAPI, HTTPException
from pydantic_core.core_schema import model_field

from pydantic_models import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn
from dataset import read_csv, validate_rows, DATA_PATH
from collections import Counter
from preprocessing import prepare_data, data_distribution, data_split, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from model import train_churn_model, evaluate_model, save_churn_model, load_churn_model
import pandas as pd


app = FastAPI()
rows = read_csv(DATA_PATH)
model_info = load_churn_model()


@app.get("/")
def read_root():
    return {
        "message": "ml churn service is running"
    }


@app.post("/predict", response_model=list[PredictionResponseChurn])
def predict(features: FeatureVectorChurn | list[FeatureVectorChurn]):
    if not isinstance(features, list):
        features = [features]

    table = pd.DataFrame([client.model_dump() for client in features])
    table = table[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    if model_info is None:
        raise HTTPException(503, "Model is not trained yet. Call POST /model/train first")

    model = model_info["model"]
    prediction = model.predict(table)
    predict_proba = model.predict_proba(table)
    result = []
    for i in range(len(features)):
        item = PredictionResponseChurn(
            churn=int(prediction[i]),
            probability_stay=float(predict_proba[i][0]),
            probability_churn=float(predict_proba[i][1])
        )
        result.append(item)
    return result


@app.get("/dataset/preview")
def preview(n: int = 5):
    if n < 1:
        raise HTTPException(400, "n must be greater than 0")
    return validate_rows(rows)[:n]


@app.get("/dataset/info")
def dataset_info():
    return {
        "rows": len(rows),
        "columns": len(rows[0]),
        "column_names": list(rows[0].keys()),
        "churn_distribution": Counter(row["churn"] for row in rows)
    }


@app.get("/dataset/split-info")
def split_info():
    x, y = prepare_data(validate_rows(rows))
    x_train, x_test, y_train, y_test = data_split(x, y)
    distribution = data_distribution(y_train, y_test)
    return {
        "train_size": len(x_train),
        "test_size": len(x_test),
        "train_churn_distribution": distribution["train"],
        "test_churn_distribution": distribution["test"]
    }


@app.post("/model/train")
def train_model(config: TrainingConfigChurn):
    global model_info

    if not rows:
        raise HTTPException(400, "No data found")
    else:
        x, y = prepare_data(validate_rows(rows))
        x_train, x_test, y_train, y_test = data_split(x, y)

        try:
            train_m = train_churn_model(x_train, y_train, config.model_type, config.hyperparameters)
        except Exception as e:
            raise HTTPException(400, f"Model training failed: {e}")

        info = {
            "model": train_m,
            "time": datetime.datetime.now().isoformat(),
            "metrics": evaluate_model(train_m, x_test, y_test),
            "model_type": config.model_type,
            "hyperparameters": config.hyperparameters,
        }

        save_churn_model(info)
        model_info = info
        return info["metrics"]


@app.get("/model/status")
def model_status():
    if model_info is None:
        return {"trained": False}
    else:
        return {
            "trained": True,
            "time": model_info["time"],
            "metrics": model_info["metrics"],
            "model_type": model_info["model_type"],
            "hyperparameters": model_info["hyperparameters"],
        }

@app.get("/model/schema")
def model_schema():
    return {
        "numeric_features": {
            feature: FeatureVectorChurn.model_fields[feature].annotation.__name__
            for feature in NUMERIC_FEATURES
        },
        "categorical_features": {
            feature: FeatureVectorChurn.model_fields[feature].annotation.__name__
            for feature in CATEGORICAL_FEATURES
        },
    }
