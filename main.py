from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}

class FeatureVectorChurn(BaseModel):
        monthly_fee: float
        usage_hours: float
        support_requests: int
        account_age_months: int
        failed_payments: int
        region: str
        device_type: str
        payment_method: str
        autopay_enabled: int


class DatasetRowChurn(BaseModel):
        monthly_fee: float
        usage_hours: float
        support_requests: int
        account_age_months: int
        failed_payments: int
        region: str
        device_type: str
        payment_method: str
        autopay_enabled: int
        churn: int

@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features

