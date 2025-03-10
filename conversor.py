import os
import tensorflow as tf

def load_keras_model(filepath):
    """
    Carrega um modelo salvo no formato .keras.
    """
    try:
        model = tf.keras.models.load_model(filepath)
        print(f"Modelo carregado de: {filepath}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def convert_to_tflite_with_selected_ops(model, output_path):
    """
    Converte um modelo Keras para o formato TensorFlow Lite (.tflite) com Selected TensorFlow Ops e salva no disco.
    """
    try:
        # Configurar o conversor
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Operações padrão do TFLite
            tf.lite.OpsSet.SELECT_TF_OPS     # Operações adicionais do TensorFlow
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_enable_resource_variables = True

        # Conversão para TFLite
        tflite_model = converter.convert()

        # Salvar o modelo convertido
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"Modelo salvo no formato TFLite em: {output_path}")
    except Exception as e:
        print(f"Erro ao converter o modelo para TFLite: {e}")

def process_directory(base_dir, output_base_dir, subfolders):
    """
    Processa pastas contendo modelos .keras, converte para .tflite e salva.
    """
    for subfolder in subfolders:
        input_dir = os.path.join(base_dir, subfolder)
        output_dir = os.path.join(output_base_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nProcessando a pasta: {subfolder}")

        for filename in os.listdir(input_dir):
            if filename.endswith(".keras"):
                input_path = os.path.join(input_dir, filename)
                model = load_keras_model(input_path)

                if model:
                    # Define o caminho de saída para o modelo .tflite
                    model_name = os.path.splitext(filename)[0]  # Remove a extensão .keras
                    output_path = os.path.join(output_dir, f"{model_name}.tflite")

                    # Converte e salva usando Selected TensorFlow Ops
                    convert_to_tflite_with_selected_ops(model, output_path)

# Exemplo de uso
if __name__ == "__main__":
    # Diretório base contendo as pastas de modelos
    base_dir = ""  # Substitua pelo caminho do diretório base (GRU, LSTM, SRNN)
    output_base_dir = "tflite_modelos"  # Diretório base para salvar os modelos convertidos
    os.makedirs(output_base_dir, exist_ok=True)

    # Subpastas para processar
    subfolders = ["GRU", "LSTM", "SRNN"]

    # Processa e converte os modelos
    process_directory(base_dir, output_base_dir, subfolders)
