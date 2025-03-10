import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from Git.dados import X_train, X_test, Y_train, Y_test, entrada, saida # Assumindo que os dados de teste estão disponíveis neste módulo


def load_tflite_model(tflite_model_path):
    """
    Carrega um modelo .tflite para uso com o TensorFlow Lite Interpreter.
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        print(f"Modelo .tflite carregado de: {tflite_model_path}")
        return interpreter
    except Exception as e:
        print(f"Erro ao carregar o modelo .tflite: {e}")
        return None


def test_tflite_model(interpreter, X_test, Y_test, output_dir, model_name):
    """
    Testa o modelo TFLite em dados de teste, calcula métricas e salva gráficos.
    """
    try:
        # Obtendo detalhes dos tensores de entrada e saída
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Forma esperada do input pelo modelo
        input_shape = input_details[0]['shape']
        print(f"Esperado input shape pelo modelo: {input_shape}")

        # Ajustar X_test ao formato esperado pelo modelo
        expected_shape = input_shape[1:]  # Exclui o batch size (1)
        if X_test.shape[1:] != tuple(expected_shape):
            raise ValueError(f"O formato do X_test ({X_test.shape}) não é compatível com o modelo.")

        # Certificar que os dados são do tipo float32
        X_test_adjusted = X_test.astype(np.float32)

        predictions = []
        for i in range(len(X_test_adjusted)):
            input_data = X_test_adjusted[i:i+1]  # Um exemplo por vez
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0])

        predictions = np.array(predictions)

        # Calcular métricas
        mse = mean_squared_error(Y_test, predictions)
        mae = mean_absolute_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)

        print(f"Test Loss (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R² Score: {r2}")

        # Salvar gráficos e métricas
        save_results(Y_test, predictions, mse, mae, r2, output_dir, model_name)

    except Exception as e:
        print(f"Erro durante o teste do modelo .tflite: {e}")


def save_results(Y_test, predictions, mse, mae, r2, output_dir, model_name):
    """
    Salva gráficos e métricas de resultados do modelo.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Salvar o gráfico de previsões vs valores reais
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, predictions, alpha=0.7, label="Previsões", color="Blue")
    plt.plot(Y_test, Y_test, color="Organge", linestyle="--", label="Valores Reais (ideal)")
    plt.title(f"Resultados do Modelo: {model_name}")
    plt.xlabel("Valores Reais")
    plt.ylabel("Previsões")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, f"{model_name}_results.png")
    plt.savefig(output_path)
    plt.close()  # Fecha o gráfico sem mostrar
    print(f"Gráfico salvo em: {output_path}")

    # Salvar métricas em um arquivo de texto
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test Loss (MSE): {mse}\n")
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"R² Score: {r2}\n")
    print(f"Métricas salvas em: {metrics_path}")


def process_and_test_tflite_models(base_dir, subfolders, X_test, Y_test):
    """
    Carrega e testa todos os modelos TFLite em subpastas específicas.
    """
    for subfolder in subfolders:
        tflite_dir = os.path.join(base_dir, subfolder)
        output_dir = os.path.join(tflite_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nTestando modelos na pasta: {tflite_dir}")

        for filename in os.listdir(tflite_dir):
            if filename.endswith(".tflite"):
                model_path = os.path.join(tflite_dir, filename)
                interpreter = load_tflite_model(model_path)

                if interpreter:
                    # Testa o modelo carregado
                    print(f"Resultados para o modelo: {filename}")
                    test_tflite_model(interpreter, X_test, Y_test, output_dir, filename)


# Exemplo de uso
if __name__ == "__main__":
    # Diretório base contendo os modelos .tflite
    base_dir = "tflite_modelos"  # Diretório onde os modelos .tflite foram salvos
    subfolders = ["GRU", "LSTM", "SRNN"]  # Subpastas que contêm os modelos .tflite

    # Testando os modelos .tflite
    process_and_test_tflite_models(base_dir, subfolders, X_train, Y_train)
