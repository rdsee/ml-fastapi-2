from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score



def train_churn_model(x_train, y_train):
    x_train_table = pd.DataFrame(x_train)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
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

