import json
import pickle


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def load_pickle(file: str, mode: str = "rb"):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a
