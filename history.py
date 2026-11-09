import json
from pathlib import Path

HISTORY_PATH = Path(__file__).parent/"artifacts"/"training_history.json"

def save_history(info, path=HISTORY_PATH):
    path.parent.mkdir(exist_ok=True)
    records = load_history(path)
    records.append(info)
    with open(path, "w") as f:
        json.dump(records, f)

def load_history(path=HISTORY_PATH):
    try:
        with open(path) as f:
            res = json.load(f)
        return res
    except FileNotFoundError:
        return []

