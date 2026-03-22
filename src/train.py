"""
train.py
--------
Pipeline de treinamento para todos os modelos RNN (SRNN / GRU / LSTM).
Unifica a lógica de main.py (regressão) e main2.py (classificação).

Uso
---
    # Classificação (padrão do projeto)
    python src/train.py

    # Regressão (modo alternativo)
    python src/train.py --mode regression
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score,
)
from tensorflow.keras.callbacks import EarlyStopping

from src.models import build_model, MODEL_REGISTRY
from src.preprocess import load_classification_data, load_regression_data

# ---------------------------------------------------------------------------
# Configurações padrão
# ---------------------------------------------------------------------------
EPOCHS       = 500
BATCH_SIZE   = 32
VAL_SPLIT    = 0.1
PATIENCE     = 10
MODEL_TYPES  = list(range(1, 16))   # 1–15
SHOW_PLOTS   = False
MODELS_DIR   = "models"
RESULTS_DIR  = "results"


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_model(model, folder: str, name: str):
    path = os.path.join(folder, f"{name}.keras")
    model.save(path)
    print(f"  [train] Modelo salvo: {path}")


def save_epoch_info(folder: str, name: str, total: int, stopped: int):
    path = os.path.join(folder, f"{name}_epochs.txt")
    with open(path, "w") as f:
        f.write(f"Épocas configuradas: {total}\n")
        if stopped > 0:
            f.write(f"EarlyStopping na época: {stopped + 1}\n")
        else:
            f.write("Treinamento completo (sem interrupção antecipada).\n")
    print(f"  [train] Info de épocas salva: {path}")


def plot_training_history(history, folder: str, name: str, show: bool = False):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"],     label="Treino",    color="red")
    plt.plot(history.history["val_loss"], label="Validação", color="blue")
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.title("Curva de aprendizado")
    plt.legend()
    path = os.path.join(folder, f"{name}_training_history.png")
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f"  [train] Histórico salvo: {path}")


def plot_predictions(Y_true, Y_pred, folder: str, name: str,
                     mode: str = "classification", show: bool = False):
    plt.figure(figsize=(12, 6))
    plt.plot(Y_true, label="Esperado",  linestyle="--", color="orange", alpha=0.7)
    plt.plot(Y_pred, label="Previsto",  color="blue",   alpha=0.7)
    plt.title("Saídas esperadas vs. previstas")
    plt.xlabel("Amostras")
    plt.ylabel("Classe" if mode == "classification" else "Valor")
    plt.legend()
    plt.grid(True)
    path = os.path.join(folder, f"{name}_plot.png")
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f"  [train] Plot salvo: {path}")


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train_model(model, X_train, Y_train, epochs: int,
                patience: int = PATIENCE, val_split: float = VAL_SPLIT):
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=val_split,
        callbacks=[es],
    )
    return history, es.stopped_epoch


# ---------------------------------------------------------------------------
# Avaliação — classificação
# ---------------------------------------------------------------------------

def evaluate_classification(model, X, Y, folder: str, name: str, show: bool = False):
    Y_pred = np.argmax(model.predict(X), axis=-1)
    acc    = accuracy_score(Y, Y_pred)
    report = classification_report(Y, Y_pred)

    print(f"  [eval] Acurácia: {acc:.4f}")

    path = os.path.join(folder, f"{name}_classification_report.txt")
    with open(path, "w") as f:
        f.write(f"Acurácia: {acc:.4f}\n\n{report}")
    print(f"  [eval] Relatório salvo: {path}")

    plot_predictions(Y, Y_pred, folder, name, mode="classification", show=show)


# ---------------------------------------------------------------------------
# Avaliação — regressão
# ---------------------------------------------------------------------------

def evaluate_regression(model, X, Y, folder: str, name: str, show: bool = False):
    Y_pred = model.predict(X).flatten()
    loss   = model.evaluate(X, Y, verbose=0)
    mae    = mean_absolute_error(Y, Y_pred)
    r2     = r2_score(Y, Y_pred)

    print(f"  [eval] Loss={loss:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    path = os.path.join(folder, f"{name}_data.txt")
    with open(path, "w") as f:
        f.write(f"Test Loss: {loss}\nMAE: {mae}\nR²: {r2}\n")

    plot_predictions(Y, Y_pred, folder, name, mode="regression", show=show)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_classification_pipeline(epochs: int = EPOCHS, model_types=MODEL_TYPES,
                                 show: bool = SHOW_PLOTS):
    print("\n=== Modo: Classificação ===")
    X_train, X_test, Y_train, Y_test, X_full, Y_full, _ = load_classification_data()

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(Y_train))

    for var in model_types:
        print(f"\n--- Modelo var={var} ({MODEL_REGISTRY[var][0]}) ---")
        model, model_name, subfamily = build_model(var, input_shape, num_classes)

        folder = os.path.join(MODELS_DIR, subfamily)
        make_dir(folder)

        history, stopped = train_model(model, X_train, Y_train, epochs)
        full_name = f"{epochs}-{model_name}"

        plot_training_history(history, folder, full_name, show)
        save_model(model, folder, full_name)
        save_epoch_info(folder, full_name, epochs, stopped)

        results_folder = os.path.join(RESULTS_DIR, "metrics", subfamily)
        make_dir(results_folder)

        evaluate_classification(model, X_test,  Y_test,  results_folder, full_name,          show)
        evaluate_classification(model, X_full,  Y_full,  results_folder, f"{full_name}_full", show)


def run_regression_pipeline(epochs: int = EPOCHS, model_types=MODEL_TYPES,
                             show: bool = SHOW_PLOTS):
    """
    Pipeline de regressão (usado originalmente com árvores de decisão e main.py).
    """
    print("\n=== Modo: Regressão ===")
    X, y = load_regression_data()

    input_shape = (1, X.shape[1])
    num_classes = 1   # saída contínua

    for var in model_types:
        print(f"\n--- Modelo var={var} ({MODEL_REGISTRY[var][0]}) ---")
        # Para regressão, recompila com MSE
        model, model_name, subfamily = build_model(var, input_shape, num_classes)
        model.compile(optimizer="adam", loss="mean_squared_error")

        folder = os.path.join(MODELS_DIR, subfamily)
        make_dir(folder)

        X_arr = X.values.reshape((X.shape[0], 1, X.shape[1]))
        history, stopped = train_model(model, X_arr, y.values, epochs)
        full_name = f"{epochs}-{model_name}"

        plot_training_history(history, folder, full_name, show)
        save_model(model, folder, full_name)
        save_epoch_info(folder, full_name, epochs, stopped)

        results_folder = os.path.join(RESULTS_DIR, "metrics", subfamily)
        make_dir(results_folder)
        evaluate_regression(model, X_arr, y.values, results_folder, full_name, show)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de treinamento RNN")
    parser.add_argument(
        "--mode", choices=["classification", "regression"],
        default="classification",
        help="Modo de treino (default: classification)",
    )
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--models",  type=int, nargs="+", default=MODEL_TYPES,
                        help="Quais variantes treinar (1–15)")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    if args.mode == "classification":
        run_classification_pipeline(args.epochs, args.models, args.show_plots)
    else:
        run_regression_pipeline(args.epochs, args.models, args.show_plots)
