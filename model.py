from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path


def train_churn_model(x_train, y_train, model_type, hyperparameters):
    x_train_table = pd.DataFrame(x_train)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    if model_type == "LogisticRegression":
        estimator = LogisticRegression(**hyperparameters)
    elif model_type == "RandomForestClassifier":
        estimator = RandomForestClassifier(**hyperparameters)
    else:
        raise ValueError("model_type must be either 'LogisticRegression' or 'RandomForestClassifier'")


    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    model.fit(x_train_table, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    x_test_table = pd.DataFrame(x_test)
    predictions = model.predict(x_test_table)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return {
        "accuracy": accuracy,
        "f1": f1,
    }

MODEL_PATH = Path(__file__).parent/"artifacts"/"churn_model.joblib"

def save_churn_model(info, path=MODEL_PATH):
    path.parent.mkdir(exist_ok=True)
    joblib.dump(info, path)

def load_churn_model(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None



