"""
convert.py
----------
Conversão de modelos para formatos embarcados:
  • Keras (.keras)  →  TensorFlow Lite (.tflite)
  • sklearn tree    →  C header (.h) via emlearn e micromlgen

Uso
---
    # Converter todos os .keras em SRNN/, GRU/, LSTM/
    python src/convert.py --task tflite

    # Gerar headers C das árvores de decisão
    python src/convert.py --task tree --tree-data data/processed/dados_normalizados_s.xlsx
"""

import os
import csv
import argparse

# ---------------------------------------------------------------------------
# TFLite
# ---------------------------------------------------------------------------

def load_keras_model(filepath: str):
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(filepath)
        print(f"  [convert] Modelo carregado: {filepath}")
        return model
    except Exception as e:
        print(f"  [convert] Erro ao carregar {filepath}: {e}")
        return None


def convert_to_tflite(model, output_path: str):
    """
    Converte um modelo Keras para .tflite com Selected TF Ops
    (necessário para modelos com LSTM/GRU/Bidirectional).
    """
    import tensorflow as tf
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_enable_resource_variables = True

        tflite_model = converter.convert()
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"  [convert] TFLite salvo: {output_path}")
    except Exception as e:
        print(f"  [convert] Erro na conversão: {e}")


def convert_all_keras_to_tflite(
    models_dir: str = "models",
    output_dir: str = "models/tflite",
    subfolders: list = None,
):
    """
    Percorre subpastas de modelos .keras e converte cada um para .tflite.
    """
    if subfolders is None:
        subfolders = ["SRNN", "GRU", "LSTM"]

    for sf in subfolders:
        input_dir  = os.path.join(models_dir, sf)
        output_sub = os.path.join(output_dir, sf)
        os.makedirs(output_sub, exist_ok=True)

        print(f"\n=== Convertendo {sf} ===")
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith(".keras"):
                continue
            model = load_keras_model(os.path.join(input_dir, fname))
            if model:
                out_name = os.path.splitext(fname)[0] + ".tflite"
                convert_to_tflite(model, os.path.join(output_sub, out_name))


# ---------------------------------------------------------------------------
# Árvores de decisão → C headers
# ---------------------------------------------------------------------------

def train_and_export_decision_trees(
    data_path: str = "data/processed/dados_normalizados_s.xlsx",
    output_dir: str = "embedded/decision_tree",
    depth_range: range = range(21, 101, 5),
    plot: bool = True,
):
    """
    Treina árvores de decisão em diferentes profundidades e exporta:
      - Header emlearn  (.h)
      - Header micromlgen (.h)
      - Plot de predição (.png)
      - Arquivo de MSE   (.csv)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error
    import emlearn
    from micromlgen import port

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "mse_norm_tree.csv")

    raw = pd.read_excel(
        data_path,
        usecols="B,C,D,E,F",
        names=["Tempo", "Externo", "Saida", "Interno", "Porta"],
    )
    data = raw.apply(pd.to_numeric, errors="coerce").dropna()
    y    = data.pop("Saida")
    X    = data

    for depth in depth_range:
        print(f"\n  [tree] Profundidade {depth}")
        reg = DecisionTreeRegressor(max_depth=depth)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        mse    = mean_squared_error(y, y_pred)

        base_name = f"decision_tree_depth{depth}"

        # emlearn header
        cmodel = emlearn.convert(reg, method="inline")
        cmodel.save(
            file=os.path.join(output_dir, f"{base_name}_emlearn.h"),
            name="decision_tree",
        )

        # micromlgen header
        c_code = port(reg)
        with open(os.path.join(output_dir, f"{base_name}_micromlgen.h"), "w") as f:
            f.write(c_code)

        # MSE log
        with open(csv_path, "a", encoding="UTF8", newline="") as f:
            csv.writer(f).writerow([mse, depth, "-"])
        print(f"  [tree] MSE={mse:.4f}")

        # Plot
        if plot:
            plt.figure(figsize=(19, 10))
            plt.plot(y_pred,           label="Previsto")
            plt.plot(y.values,         label="Real", linestyle=":")
            plt.legend()
            plot_path = os.path.join(
                output_dir, f"tree_depth{depth}_mse{round(mse, 4)}.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"  [tree] Plot salvo: {plot_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversão de modelos")
    parser.add_argument(
        "--task", choices=["tflite", "tree"], required=True,
        help="Tarefa: 'tflite' converte keras→tflite; 'tree' exporta árvores em C",
    )
    parser.add_argument(
        "--models-dir",  default="models",
        help="Diretório raiz dos modelos .keras (default: models/)",
    )
    parser.add_argument(
        "--tflite-dir",  default="models/tflite",
        help="Diretório de saída dos .tflite (default: models/tflite/)",
    )
    parser.add_argument(
        "--tree-data",
        default="data/processed/dados_normalizados_s.xlsx",
        help="Arquivo de dados para treino das árvores",
    )
    parser.add_argument(
        "--tree-output", default="embedded/decision_tree",
        help="Diretório de saída dos headers C",
    )
    parser.add_argument(
        "--depth-min",  type=int, default=21)
    parser.add_argument(
        "--depth-max",  type=int, default=101)
    parser.add_argument(
        "--depth-step", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.task == "tflite":
        convert_all_keras_to_tflite(
            models_dir=args.models_dir,
            output_dir=args.tflite_dir,
        )
    else:
        train_and_export_decision_trees(
            data_path=args.tree_data,
            output_dir=args.tree_output,
            depth_range=range(args.depth_min, args.depth_max, args.depth_step),
            plot=not args.no_plot,
        )
