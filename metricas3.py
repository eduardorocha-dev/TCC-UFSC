import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# Carregar e preparar os dados
file_name = "dados_concatenados_norma2.xlsx"
raw_data = pd.read_excel(file_name, usecols="A,B,C,D", names=['Tempo', 'ARexterno', 'ARinterno', 'Saida'])

data = raw_data.copy()

# Separar entradas e saídas
entrada = data.drop(columns=['Saida'])  # Entradas
saida = data['Saida']  # Saídas

# Dividir os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X_train, X_test, Y_train, Y_test = train_test_split(entrada, saida, test_size=0.3, random_state=42)

# Codificar os rótulos de saída (caso sejam categóricos)
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
Y_test = label_encoder.transform(Y_test)

# Ajustar os dados de entrada para o formato esperado pelas RNNs [samples, time_steps, features]
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

Y_extra = label_encoder.transform(saida)  # Transformar os rótulos originais para Y_extra
X_extra = entrada.values.reshape((entrada.shape[0], 1, entrada.shape[1]))

# Função para carregar modelo .tflite
def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Função para fazer previsões usando o modelo TFLite
def predict_tflite(interpreter, input_details, output_details, data):
    interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Caminho para o modelo TFLite
tflite_model_path = "tflite_modelos\LSTM\LSTM-Dropout.tflite"

# Carregar o modelo
interpreter, input_details, output_details = load_tflite_model(tflite_model_path)

# Fazer previsões
y_pred = np.array([predict_tflite(interpreter, input_details, output_details, x.reshape(1, *x.shape)) for x in X_extra]).squeeze()
y_pred_labels = np.argmax(y_pred, axis=1)  # Obter rótulos das previsões caso sejam probabilidades

# Cálculo das métricas
accuracy = accuracy_score(Y_extra, y_pred_labels)
classification_report_text = classification_report(Y_extra, y_pred_labels)

# Número ideal de épocas
ideal_epochs = 161

# Salvar métricas em um arquivo de texto
with open("metrics.txt", "w") as f:
    f.write(f"Acurácia: {accuracy:.4f}\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(classification_report_text)
    f.write(f"\nNúmero ideal de épocas: {ideal_epochs}\n")

# Plotar resultados
plt.figure(figsize=(10, 6))
plt.plot(Y_extra, label="Saída Esperada", linestyle="--", color='Orange')
plt.plot(y_pred_labels, label="Saída Prevista", color='Blue')
plt.xlabel("Amostras")
plt.ylabel("Valores")
plt.title("Comparação entre Saída Esperada e Prevista")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output_comparison.png")
#plt.show()
