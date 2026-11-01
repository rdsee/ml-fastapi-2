from collections import Counter

from sklearn.model_selection import train_test_split
NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled"
]

def prepare_data(rows):
    x = []
    y = []
    for row in rows:
        x_dict = {}
        d = row.model_dump()
        y.append(d["churn"])
        for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            x_dict[feature] = d[feature]
        x.append(x_dict)
    return x, y

def data_split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test

def data_distribution(y_train, y_test):
    return {
        "train": Counter(y_train),
        "test": Counter(y_test)
    }



