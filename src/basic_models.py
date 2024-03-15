import pickle
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (TextVectorization, Embedding, Dense, Flatten, Dropout,
                                     BatchNormalization, Conv1D, MaxPooling1D, GRU)


with open(Path(__file__).parent.parent / 'data' / 'vocab.pkl', 'rb') as file:
    VOCAB = pickle.load(file)


def create_model_1(vec_len):
    model = Sequential([
        TextVectorization(
          max_tokens=len(VOCAB) + 2,
          standardize=None,
          vocabulary=VOCAB,
          split='whitespace',
          output_mode='int',
          output_sequence_length=vec_len
          ),
        Embedding(input_dim=len(VOCAB) + 2, output_dim=128, input_length=vec_len),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(.5),
        Dense(32, activation='relu'),
        Dropout(.3),
        Dense(6, activation='softmax')
    ])
    return model


if __name__ == '__main__':
    print(create_model_1(34))
