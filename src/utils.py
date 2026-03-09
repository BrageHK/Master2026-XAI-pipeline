from pathlib import Path
import pickle

def load_plans(plans_file: str | Path) -> dict:
    with open(plans_file, "rb") as f:
        return pickle.load(f)