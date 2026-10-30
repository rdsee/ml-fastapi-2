from fastapi import FastAPI, HTTPException
from models import FeatureVectorChurn
from dataset import read_csv, validate_rows, DATA_PATH
from collections import Counter


app = FastAPI()
rows = read_csv(DATA_PATH)

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features

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
