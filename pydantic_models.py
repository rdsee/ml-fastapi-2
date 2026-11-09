from typing import Literal
from pydantic import BaseModel

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
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "monthly_fee": 6.9,
                    "usage_hours": 20.1,
                    "support_requests": 3,
                    "account_age_months": 11,
                    "failed_payments": 0,
                    "region": "europe",
                    "device_type": "desktop",
                    "payment_method": "paypal",
                    "autopay_enabled": 1
                }
            ]
        }
    }



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


class PredictionResponseChurn(BaseModel):
    churn: int
    probability_stay: float
    probability_churn: float
    model_config = {
        "json_schema_extra": {
            "examples": [
                  {
                    "churn": 0,
                    "probability_stay": 0.8596788779059112,
                    "probability_churn": 0.14032112209408873
                  }
            ]
        }
    }

class TrainingConfigChurn(BaseModel):
    model_type: Literal["LogisticRegression", "RandomForestClassifier"]
    hyperparameters: dict


class ErrorConfig(BaseModel):
    code: str
    message: str
    details: list | None = None


