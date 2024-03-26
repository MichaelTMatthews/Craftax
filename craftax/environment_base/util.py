import pickle
import bz2
from typing import Any


def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title, "w") as f:
        pickle.dump(data, f)


def load_compressed_pickle(file: str):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data
