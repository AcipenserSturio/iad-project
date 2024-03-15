import numpy as np
import pandas as pd
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (TextVectorization, Embedding, Dense, Flatten, Dropout,
                                     BatchNormalization, Conv1D, MaxPooling1D, GRU)


class SymbioticNN:
    def __init__(self, model1: Sequential, model2: Sequential, m1w: float = 0.5):
        for model in (model1, model2):
            if not isinstance(model, Sequential):
                raise TypeError(f"{model} expected to be an instance of keras.Sequential class, "
                                f"but it's actual class is {type(model)}.")

        self._balance = m1w
        self._model1 = model1
        self._model2 = model2
        self._classes_mapping = {
            0: 'NO CYBERBULLYING',
            1: 'GENDER',
            2: 'RELIGION',
            3: 'OTHER CYBERBULLYING',
            4: 'AGE',
            5: 'ETHNICITY'
        }

    def infer(self, preprocessed_text: str) -> str:
        wrapped_text = pd.Series([preprocessed_text])
        pred1 = self._model1(wrapped_text, training=False)
        pred2 = self._model2(wrapped_text, training=False)
        pred = np.log(pred1 * self._balance + pred2 * (1 - self._balance))
        prediction = np.argmax(pred, axis=1)

        return self._classes_mapping.get(*prediction)

    @property
    def model1(self):
        return self._model1

    @property
    def model2(self):
        return self._model2
