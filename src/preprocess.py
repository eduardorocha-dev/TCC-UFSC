"""
preprocess.py
-------------
Carregamento, limpeza e preparação dos dados para treino/avaliação.
Suporta dois modos:
  - 'regression'     : dados do arquivo de regressão (arvore + main.py original)
  - 'classification' : dados do arquivo de classificação (main2.py / metricas3.py)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Caminhos padrão (podem ser sobrescritos via config.yaml)
# ---------------------------------------------------------------------------
REGRESSION_FILE    = "data/processed/dados_normalizados_s.xlsx"
CLASSIFICATION_FILE = "data/processed/dados_concatenados_norma2.xlsx"

# ---------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------

def load_regression_data(file_path: str = REGRESSION_FILE):
    """
    Carrega os dados de regressão (séries temporais com sensor de temperatura).

    Retorna
    -------
    X : pd.DataFrame  — features (Tempo, Externo, Interno, Porta)
    y : pd.Series     — alvo (Saida)
    """
    raw = pd.read_excel(
        file_path,
        usecols="B,C,D,E,F",
        names=["Tempo", "Externo", "Saida", "Interno", "Porta"],
    )
    data = raw.apply(pd.to_numeric, errors="coerce").dropna()
    print(f"[preprocess] NaN restantes: {data.isnull().sum().sum()}")

    y = data.pop("Saida")
    X = data
    return X, y


# ---------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------

def load_classification_data(
    file_path: str = CLASSIFICATION_FILE,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    Carrega e divide os dados de classificação para RNNs.

    Retorna
    -------
    X_train, X_test, Y_train, Y_test : arrays no formato (samples, 1, features)
    X_full, Y_full                   : dataset completo na ordem original
    label_encoder                    : LabelEncoder já ajustado
    """
    raw = pd.read_excel(
        file_path,
        usecols="A,B,C,D",
        names=["Tempo", "ARexterno", "ARinterno", "Saida"],
    )
    data = raw.copy()
    entrada = data.drop(columns=["Saida"])
    saida   = data["Saida"]

    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        entrada, saida, test_size=test_size, random_state=random_state
    )

    le = LabelEncoder()
    Y_train = le.fit_transform(y_train_s)
    Y_test  = le.transform(y_test_s)
    Y_full  = le.transform(saida)

    # Reshape para (samples, timesteps=1, features)
    def reshape(df):
        arr = df.values if hasattr(df, "values") else df
        return arr.reshape((arr.shape[0], 1, arr.shape[1]))

    X_train = reshape(X_train_df)
    X_test  = reshape(X_test_df)
    X_full  = reshape(entrada)

    print(
        f"[preprocess] Train: {X_train.shape} | Test: {X_test.shape} | Full: {X_full.shape}"
    )
    print(f"[preprocess] Classes: {le.classes_}")

    return X_train, X_test, Y_train, Y_test, X_full, Y_full, le
