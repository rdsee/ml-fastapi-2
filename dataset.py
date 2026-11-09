import csv
from pathlib import Path
from pydantic import ValidationError
from pydantic_models import DatasetRowChurn
from errors import ChurnServiceError

DATA_PATH = Path(__file__).parent/"data"/"churn_dataset.csv"

def read_csv(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        data_list = []
        for row in reader:
            data_list.append(row)
        return data_list


from pydantic import ValidationError

def validate_rows(rows):
    result = []
    for number, row in enumerate(rows, start=1):
        try:
            result.append(DatasetRowChurn.model_validate(row))
        except ValidationError as e:
            raise ChurnServiceError(
                500,
                "invalid_dataset_row",
                f"Row {number} in the dataset is invalid",
                [{"field": err["loc"][-1], "problem": err["msg"]} for err in e.errors()],
            )
    return result
