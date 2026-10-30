import csv
from pathlib import Path
from models import DatasetRowChurn

DATA_PATH = Path(__file__).parent/"data"/"churn_dataset.csv"

def read_csv(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        data_list = []
        for row in reader:
            data_list.append(row)
        return data_list

def validate_rows(rows):
    result = []
    for row in rows:
        result.append(DatasetRowChurn.model_validate(row))
    return result

