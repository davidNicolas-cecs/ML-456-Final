import numpy as np

from tensorflow.keras.utils import to_categorical  # type: ignore


translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}


label_dict = {
    "cane": 0,
    "cavallo": 1,
    "elefante": 2,
    "farfalla": 3,
    "gallina": 4,
    "gatto": 5,
    "mucca": 6,
    "pecora": 7,
    "scoiattolo": 8,
    "ragno": 9,
}


def hot_encode(y):
    return to_categorical(y, num_classes=10)  


def encode(y):
    return np.array([label_dict[label] for label in y])
