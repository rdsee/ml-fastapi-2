import datetime
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from sklearn.metrics import roc_auc_score
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from errors import ChurnServiceError
from history import save_history, load_history
from pydantic_models import FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn, ErrorConfig
from dataset import read_csv, validate_rows, DATA_PATH
from collections import Counter
from preprocessing import prepare_data, data_distribution, data_split, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from model import train_churn_model, evaluate_model, save_churn_model, load_churn_model
import pandas as pd


app = FastAPI()
rows = read_csv(DATA_PATH)
model_info = load_churn_model()

def error_example(description, code, message, details=None):
    return {
        "model": ErrorConfig,
        "description": description,
        "content": {"application/json": {"example": {"code": code, "message": message, "details": details}}},
    }


PREDICT_ERRORS = {
    422: error_example("Invalid request data", "validation_error", "Request data is invalid",
                       [{"field": "region", "problem": "Field required"}]),
    503: error_example("Model is not trained", "model_not_trained",
                       "Model is not trained yet. Call POST /model/train first"),
    500: error_example("Unexpected server error", "internal_error", "Internal server error"),
}

TRAIN_ERRORS = {
    400: error_example("No data or training failed", "model_error",
                       "Model training failed: LogisticRegression.__init__() got an unexpected keyword argument 'foo'"
                       ),
    422: error_example("Invalid model_type", "validation_error", "Request data is invalid",
                       [{"field": "model_type",
                         "problem": "Input should be 'LogisticRegression' or 'RandomForestClassifier'"}]
                       ),
    500: error_example("Invalid row in dataset", "invalid_dataset_row", "Row 2 in the dataset is invalid",
                       [{"field": "monthly_fee", "problem": "Input should be a valid number, unable to parse string as a number"}]
                       )
}


@app.exception_handler(Exception)
def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorConfig(code='internal_error', message='Internal server error', details=None).model_dump()
    )

@app.exception_handler(RequestValidationError)
def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [
        {"field": i["loc"][-1], "problem": i["msg"]}
        for i in exc.errors()
        if i["type"] != "list_type"
    ]

    return JSONResponse(
        status_code=422,
        content=ErrorConfig(code="validation_error", message="Request data is invalid", details=details).model_dump()
    )

@app.exception_handler(StarletteHTTPException)
def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorConfig(code="http_error", message=exc.detail, details=None).model_dump()
    )

@app.exception_handler(ChurnServiceError)
def churn_exception_handler(request: Request, exc: ChurnServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorConfig(code=exc.code, message=exc.message, details=exc.details).model_dump()
    )

def data_test():
    if not rows:
        raise ChurnServiceError(400, "data_error", "No data found")

@app.get("/")
def read_root():
    return {
        "message": "ml churn service is running"
    }


@app.post("/predict", response_model=list[PredictionResponseChurn], responses=PREDICT_ERRORS)
def predict(features: FeatureVectorChurn | list[FeatureVectorChurn]):
    if not isinstance(features, list):
        features = [features]

    table = pd.DataFrame([client.model_dump() for client in features])
    table = table[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    if model_info is None:
        raise ChurnServiceError(503, "model_not_trained", "Model is not trained yet. Call POST /model/train first")

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

    if len(result) > 9:
        raise Exception(FeatureVectorChurn)

    return result


@app.get("/dataset/preview")
def preview(n: int = 5):
    if n < 1:
        raise ChurnServiceError(400, "index_error", "n must be greater than 0")
    return validate_rows(rows)[:n]


@app.get("/dataset/info")
def dataset_info():
    data_test()
    return {
        "rows": len(rows),
        "columns": len(rows[0]),
        "column_names": list(rows[0].keys()),
        "churn_distribution": Counter(row["churn"] for row in rows)
    }


@app.get("/dataset/split-info")
def split_info():
    data_test()
    x, y = prepare_data(validate_rows(rows))
    x_train, x_test, y_train, y_test = data_split(x, y)
    distribution = data_distribution(y_train, y_test)
    return {
        "train_size": len(x_train),
        "test_size": len(x_test),
        "train_churn_distribution": distribution["train"],
        "test_churn_distribution": distribution["test"]
    }


@app.post("/model/train", responses=TRAIN_ERRORS)
def train_model(config: TrainingConfigChurn):
    global model_info

    if not rows:
        raise ChurnServiceError(400, "data_error", "No data found")
    else:
        x, y = prepare_data(validate_rows(rows))
        x_train, x_test, y_train, y_test = data_split(x, y)

        try:
            train_m = train_churn_model(x_train, y_train, config.model_type, config.hyperparameters)
        except Exception as e:
            raise ChurnServiceError(400, "model_error",f"Model training failed: {e}")

        info = {
            "model": train_m,
            "time": datetime.datetime.now().isoformat(),
            "metrics": evaluate_model(train_m, x_test, y_test),
            "model_type": config.model_type,
            "hyperparameters": config.hyperparameters,
        }

        history_info = {
            "time": info["time"],
            "model_type": info["model_type"],
            "hyperparameters": info["hyperparameters"],
            "metrics": info["metrics"],
        }

        save_churn_model(info)
        model_info = info
        save_history(history_info)

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

@app.get("/model/metrics")
def model_metrics(n: int = 5, model_type: Literal["LogisticRegression", "RandomForestClassifier"] | None = None):
    if n < 1:
        raise ChurnServiceError(400, "invalid_limit", "n must be greater than 0")

    history = load_history()

    if not history:
        raise ChurnServiceError(404, "history_not_found", "No data in history")

    if model_type:
        history = [i for i in history if i["model_type"] == model_type]

    if history:
        best = history[0]
        for item in history:
            if item["metrics"]["roc_auc"] > best["metrics"]["roc_auc"]:
                best = item

        worst = history[-1]
        for item in history:
            if item["metrics"]["roc_auc"] < worst["metrics"]["roc_auc"]:
                worst = item


        return {
            "latest": history[-1],
            "history": history[-n:],
            "best": {
                "model_type": best["model_type"],
                "hyperparameters": best["hyperparameters"],
                "metrics": best["metrics"],
            },
            "worst": {
                "model_type": worst["model_type"],
                "hyperparameters": worst["hyperparameters"],
                "metrics": worst["metrics"],
            }
        }
    else:
        raise ChurnServiceError(404, "model_not_found", "No records for this model")










