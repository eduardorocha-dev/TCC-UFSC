"""
models.py
---------
Definição de todos os modelos de redes neurais recorrentes utilizados no projeto.
Suporta três famílias: SRNN, GRU, LSTM — cada uma com 5 variantes:
  Base, Deep, Dropout, Bidirectional, Large.

Uso
---
    from src.models import build_model
    model, name, folder = build_model(var=1, input_shape=(1, 3), num_classes=4)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN, GRU, LSTM,
    Dense, Dropout, Bidirectional,
)

# ---------------------------------------------------------------------------
# Mapa de variantes
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # --- SRNN ---
    1:  ("SRN",               "SRNN"),
    2:  ("SRNN-Deep",         "SRNN"),
    3:  ("SRNN-Dropout",      "SRNN"),
    4:  ("SRNN-Bidirectional","SRNN"),
    5:  ("SRNN-Large",        "SRNN"),
    # --- GRU ---
    6:  ("GRU",               "GRU"),
    7:  ("GRU-Deep",          "GRU"),
    8:  ("GRU-Dropout",       "GRU"),
    9:  ("GRU-Bidirectional", "GRU"),
    10: ("GRU-Large",         "GRU"),
    # --- LSTM ---
    11: ("LSTM",              "LSTM"),
    12: ("LSTM-Deep",         "LSTM"),
    13: ("LSTM-Dropout",      "LSTM"),
    14: ("LSTM-Bidirectional","LSTM"),
    15: ("LSTM-Large",        "LSTM"),
}


def _compile(model, optimizer="adam", loss="sparse_categorical_crossentropy"):
    model.compile(optimizer=optimizer, loss=loss)
    return model


def build_model(var: int, input_shape: tuple, num_classes: int):
    """
    Constrói e compila um modelo RNN de acordo com o identificador `var`.

    Parâmetros
    ----------
    var         : int   — identificador do modelo (1–15)
    input_shape : tuple — forma da entrada sem batch, ex.: (1, 3)
    num_classes : int   — número de classes de saída

    Retorna
    -------
    (model, model_name, subfolder)
    """
    if var not in MODEL_REGISTRY:
        raise ValueError(f"'var' deve estar entre 1 e 15. Recebido: {var}")

    name, folder = MODEL_REGISTRY[var]
    out = num_classes
    act = "softmax"
    opt = "adam"
    loss = "sparse_categorical_crossentropy"

    # ------------------------------------------------------------------ SRNN
    if var == 1:
        model = Sequential([
            SimpleRNN(50, input_shape=input_shape, activation="relu"),
            Dense(out, activation=act),
        ])

    elif var == 2:
        model = Sequential()
        model.add(SimpleRNN(50, input_shape=input_shape, activation="relu", return_sequences=True))
        for _ in range(4):
            model.add(SimpleRNN(50, activation="relu", return_sequences=True))
        model.add(SimpleRNN(50, activation="relu"))
        model.add(Dense(out, activation=act))

    elif var == 3:
        model = Sequential([
            SimpleRNN(50, input_shape=input_shape, activation="relu", return_sequences=True),
            Dropout(0.2),
            SimpleRNN(50, activation="relu"),
            Dense(out, activation=act),
        ])

    elif var == 4:
        model = Sequential([
            Bidirectional(SimpleRNN(50, activation="relu"), input_shape=input_shape),
            Dense(out, activation=act),
        ])

    elif var == 5:
        model = Sequential([
            SimpleRNN(100, input_shape=input_shape, activation="relu"),
            Dense(out, activation=act),
        ])

    # ------------------------------------------------------------------- GRU
    elif var == 6:
        model = Sequential([
            GRU(50, input_shape=input_shape, activation="relu"),
            Dense(out, activation=act),
        ])

    elif var == 7:
        model = Sequential()
        model.add(GRU(50, input_shape=input_shape, activation="relu", return_sequences=True))
        for _ in range(2):
            model.add(GRU(50, activation="relu", return_sequences=True))
        model.add(GRU(50, activation="relu"))
        model.add(Dense(out, activation=act))

    elif var == 8:
        model = Sequential([
            GRU(50, input_shape=input_shape, activation="relu", return_sequences=True),
            Dropout(0.2),
            GRU(50, activation="relu"),
            Dense(out, activation=act),
        ])

    elif var == 9:
        model = Sequential([
            Bidirectional(GRU(50, activation="relu"), input_shape=input_shape),
            Dense(out, activation=act),
        ])

    elif var == 10:
        model = Sequential([
            GRU(100, input_shape=input_shape, activation="relu"),
            Dense(out, activation=act),
        ])

    # ------------------------------------------------------------------ LSTM
    elif var == 11:
        model = Sequential([
            LSTM(50, input_shape=input_shape, activation="relu"),
            Dense(out, activation=act),
        ])

    elif var == 12:
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape, activation="relu", return_sequences=True))
        for _ in range(2):
            model.add(LSTM(50, activation="relu", return_sequences=True))
        model.add(LSTM(50, activation="relu"))
        model.add(Dense(out, activation=act))

    elif var == 13:
        model = Sequential([
            LSTM(50, input_shape=input_shape, activation="relu", return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation="relu"),
            Dense(out, activation=act),
        ])

    elif var == 14:
        model = Sequential([
            Bidirectional(LSTM(50, activation="relu"), input_shape=input_shape),
            Dense(out, activation=act),
        ])

    elif var == 15:
        model = Sequential([
            LSTM(100, input_shape=input_shape, activation="relu"),
            Dense(out, activation=act),
        ])

    return _compile(model, opt, loss), name, folder
