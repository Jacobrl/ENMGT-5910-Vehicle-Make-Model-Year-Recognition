import json
from pathlib import Path


def build_metadata(car_info, damage_info=None, cost_info=None):
    if damage_info is None:
        damage_info = []

    if cost_info is None:
        cost_info = {"usd": None, "mxn": None}

    metadata = {
        "car": car_info,
        "damage": damage_info,
        "cost_estimate": cost_info
    }

    return metadata

def save_metadata(metadata, output_path):
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)
