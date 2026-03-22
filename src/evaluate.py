"""
evaluate.py
-----------
Avaliação de modelos treinados (.keras e .tflite).
Unifica metricas.py, metricas2.py e metricas3.py.

Uso
---
    # Avaliar todos os .keras
    python src/evaluate.py --format keras

    # Avaliar todos os .tflite
    python src/evaluate.py --format tflite

    # Avaliar um único modelo tflite específico
    python src/evaluate.py --format tflite --model models/tflite/LSTM/LSTM-Dropout.tflite
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score, mean_squared_error,
)

from src.preprocess import load_classification_data

# ---------------------------------------------------------------------------
# Configurações padrão
# ---------------------------------------------------------------------------
MODELS_DIR  = "models"
RESULTS_DIR = "results"
SUBFOLDERS  = ["SRNN", "GRU", "LSTM"]


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_comparison(Y_true, Y_pred, folder: str, name: str, show: bool = False):
    plt.figure(figsize=(12, 6))
    plt.plot(Y_true,  label="Esperado", linestyle="--", color="orange", alpha=0.7)
    plt.plot(Y_pred,  label="Previsto", color="blue",   alpha=0.7)
    plt.title(f"Comparação: {name}")
    plt.xlabel("Amostras")
    plt.ylabel("Classe")
    plt.legend()
    plt.grid(True)
    path = os.path.join(folder, f"{name}_comparison.png")
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f"  [eval] Plot salvo: {path}")


def _save_report(folder: str, name: str, acc: float, report: str,
                 extra: str = ""):
    path = os.path.join(folder, f"{name}_classification_report.txt")
    with open(path, "w") as f:
        f.write(f"Acurácia: {acc:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(report)
        if extra:
            f.write(f"\n{extra}")
    print(f"  [eval] Relatório salvo: {path}")


def _save_regression_metrics(folder: str, name: str,
                              mse: float, mae: float, r2: float):
    path = os.path.join(folder, f"{name}_metrics.txt")
    with open(path, "w") as f:
        f.write(f"MSE : {mse:.6f}\n")
        f.write(f"MAE : {mae:.6f}\n")
        f.write(f"R²  : {r2:.6f}\n")
    print(f"  [eval] Métricas salvas: {path}")


# ---------------------------------------------------------------------------
# Keras — avaliação
# ---------------------------------------------------------------------------

def evaluate_keras_model(filepath: str, X, Y, output_dir: str,
                         show: bool = False):
    """Avalia um modelo .keras e salva métricas + gráfico."""
    try:
        model = load_model(filepath)
    except Exception as e:
        print(f"  [eval] Erro ao carregar {filepath}: {e}")
        return

    name = os.path.splitext(os.path.basename(filepath))[0]
    make_dir(output_dir)

    Y_pred = np.argmax(model.predict(X), axis=-1)
    acc    = accuracy_score(Y, Y_pred)
    report = classification_report(Y, Y_pred)

    print(f"\n  Modelo : {name}")
    print(f"  Acurácia: {acc:.4f}")

    _save_report(output_dir, name, acc, report)
    _plot_comparison(Y, Y_pred, output_dir, name, show)


def evaluate_all_keras(subfolders=SUBFOLDERS, show: bool = False):
    """
    Itera sobre todas as subpastas de modelos .keras e avalia cada um.
    Equivale a metricas.py.
    """
    _, X_test, _, Y_test, X_full, Y_full, _ = load_classification_data()

    for sf in subfolders:
        model_dir  = os.path.join(MODELS_DIR, sf)
        output_dir = os.path.join(RESULTS_DIR, "metrics", sf)
        make_dir(output_dir)
        print(f"\n=== Avaliando .keras em: {model_dir} ===")

        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".keras"):
                continue
            fpath = os.path.join(model_dir, fname)
            evaluate_keras_model(fpath, X_test, Y_test, output_dir, show)
            evaluate_keras_model(fpath, X_full, Y_full, output_dir, show)


# ---------------------------------------------------------------------------
# TFLite — avaliação
# ---------------------------------------------------------------------------

def _load_tflite(path: str):
    try:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        print(f"  [tflite] Carregado: {path}")
        return interp
    except Exception as e:
        print(f"  [tflite] Erro ao carregar {path}: {e}")
        return None


def _predict_tflite(interp, X: np.ndarray) -> np.ndarray:
    """Executa inferência sample-a-sample e retorna array de predições."""
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()
    preds   = []
    X_f32   = X.astype(np.float32)

    for i in range(len(X_f32)):
        interp.set_tensor(in_det[0]["index"], X_f32[i : i + 1])
        interp.invoke()
        preds.append(interp.get_tensor(out_det[0]["index"])[0])
    return np.array(preds)


def evaluate_tflite_model(filepath: str, X, Y, output_dir: str,
                          show: bool = False):
    """
    Avalia um modelo .tflite (classificação).
    Equivale a metricas2.py + metricas3.py.
    """
    interp = _load_tflite(filepath)
    if interp is None:
        return

    name = os.path.splitext(os.path.basename(filepath))[0]
    make_dir(output_dir)

    raw    = _predict_tflite(interp, X)
    Y_pred = np.argmax(raw, axis=1) if raw.ndim > 1 else raw.astype(int)

    acc    = accuracy_score(Y, Y_pred)
    report = classification_report(Y, Y_pred)

    print(f"\n  Modelo : {name}")
    print(f"  Acurácia: {acc:.4f}")

    _save_report(output_dir, name, acc, report)
    _plot_comparison(Y, Y_pred, output_dir, name, show)


def evaluate_all_tflite(subfolders=SUBFOLDERS, show: bool = False):
    """
    Itera sobre todas as subpastas de modelos .tflite e avalia cada um.
    """
    _, X_test, _, Y_test, X_full, Y_full, _ = load_classification_data()

    for sf in subfolders:
        tflite_dir = os.path.join(MODELS_DIR, "tflite", sf)
        output_dir = os.path.join(RESULTS_DIR, "metrics", f"tflite_{sf}")
        make_dir(output_dir)
        print(f"\n=== Avaliando .tflite em: {tflite_dir} ===")

        for fname in sorted(os.listdir(tflite_dir)):
            if not fname.endswith(".tflite"):
                continue
            fpath = os.path.join(tflite_dir, fname)
            print(f"\n  -- Conjunto de teste --")
            evaluate_tflite_model(fpath, X_test, Y_test, output_dir, show)
            print(f"\n  -- Conjunto completo --")
            evaluate_tflite_model(fpath, X_full, Y_full, output_dir, show)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de modelos")
    parser.add_argument(
        "--format", choices=["keras", "tflite"], default="keras",
        help="Formato dos modelos a avaliar",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Caminho para um modelo específico (opcional)",
    )
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    if args.model:
        # Avalia apenas o modelo informado
        _, X_test, _, Y_test, X_full, Y_full, _ = load_classification_data()
        out = os.path.join(RESULTS_DIR, "metrics", "single")

        if args.format == "keras":
            evaluate_keras_model(args.model, X_test,  Y_test,  out, args.show_plots)
            evaluate_keras_model(args.model, X_full,  Y_full,  out, args.show_plots)
        else:
            evaluate_tflite_model(args.model, X_test,  Y_test,  out, args.show_plots)
            evaluate_tflite_model(args.model, X_full,  Y_full,  out, args.show_plots)
    else:
        if args.format == "keras":
            evaluate_all_keras(show=args.show_plots)
        else:
            evaluate_all_tflite(show=args.show_plots)
