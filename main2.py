import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, GRU, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Carregar e preparar os dados
file_name = "dados_concatenados_norma2.xlsx"
raw_data = pd.read_excel(file_name, usecols="A,B,C,D", names=['Tempo', 'ARexterno', 'ARinterno', 'Saida'])

data = raw_data.copy()

# Separar entradas e saídas
entrada = data.drop(columns=['Saida'])  # Entradas
saida = data['Saida']  # Saídas

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(entrada, saida, test_size=0.3, random_state=42)


# Codificar os rótulos de saída (caso sejam categóricos)
label_encoder = LabelEncoder()

Y_train = label_encoder.fit_transform(Y_train)
Y_test = label_encoder.transform(Y_test)

# Ajustar os dados de entrada para o formato esperado pelas RNNs [samples, time_steps, features]
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Função para criar os modelos
def Modelos(var, input_shape, num_classes):
    """
    Gera um modelo sequencial com base no tipo especificado em `var`.

    Parâmetros:
    - var (int): Identificador do tipo de modelo.
    - input_shape (tuple): Forma da entrada no modelo (timesteps, features).
    - num_classes (int): Número de classes ou unidades na camada de saída.

    Retorna:
    - tuple: (modelo, nome do modelo, pasta do modelo)
    """
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    activation_output = 'softmax'

    if var == 1:  # Simple RNN básico
        model = Sequential([
            SimpleRNN(units=50, input_shape=input_shape, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model, 'SRN', 'SRNN/'

    elif var == 2:  # Simple RNN com várias camadas
        model = Sequential()
        for _ in range(5):
            model.add(SimpleRNN(units=50, activation="relu", return_sequences=True))
        model.add(SimpleRNN(units=50, activation="relu", return_sequences=False))
        model.add(Dense(units=num_classes, activation=activation_output))
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'SRNN-Deep', 'SRNN/'

    elif var == 3:  # Simple RNN com Dropout
        model = Sequential([
            SimpleRNN(units=50, input_shape=input_shape, activation="relu", return_sequences=True),
            Dropout(0.2),
            SimpleRNN(units=50, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'SRNN-Dropout', 'SRNN/'

    elif var == 4:  # Bidirectional Simple RNN
        model = Sequential([
            Bidirectional(SimpleRNN(units=50, activation="relu", return_sequences=False), input_shape=input_shape),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'SRNN-Bidirectional', 'SRNN/'

    elif var == 5:  # Simple RNN com tamanhos de camada diferentes
        model = Sequential([
            SimpleRNN(units=100, input_shape=input_shape, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'SRNN-Large', 'SRNN/'

    elif var == 6:  # GRU básico
        model = Sequential([
            GRU(units=50, input_shape=input_shape, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'GRU', 'GRU/'

    elif var == 7:  # GRU com várias camadas
        model = Sequential()
        for _ in range(3):
            model.add(GRU(units=50, activation="relu", return_sequences=True))
        model.add(GRU(units=50, activation="relu", return_sequences=False))
        model.add(Dense(units=num_classes, activation=activation_output))
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'GRU-Deep', 'GRU/'

    elif var == 8:  # GRU com Dropout
        model = Sequential([
            GRU(units=50, input_shape=input_shape, activation="relu", return_sequences=True),
            Dropout(0.2),
            GRU(units=50, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'GRU-Dropout', 'GRU/'

    elif var == 9:  # Bidirectional GRU
        model = Sequential([
            Bidirectional(GRU(units=50, activation="relu", return_sequences=False), input_shape=input_shape),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'GRU-Bidirectional', 'GRU/'

    elif var == 10:  # GRU com tamanhos de camada diferentes
        model = Sequential([
            GRU(units=100, input_shape=input_shape, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'GRU-Large', 'GRU/'

    elif var == 11:  # LSTM básico
        model = Sequential([
            LSTM(units=50, input_shape=input_shape, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'LSTM', 'LSTM/'

    elif var == 12:  # LSTM com várias camadas
        model = Sequential()
        for _ in range(3):
            model.add(LSTM(units=50, activation="relu", return_sequences=True))
        model.add(LSTM(units=50, activation="relu", return_sequences=False))
        model.add(Dense(units=num_classes, activation=activation_output))
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'LSTM-Deep', 'LSTM/'

    elif var == 13:  # LSTM com Dropout
        model = Sequential([
            LSTM(units=50, input_shape=input_shape, activation="relu", return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'LSTM-Dropout', 'LSTM/'

    elif var == 14:  # Bidirectional LSTM
        model = Sequential([
            Bidirectional(LSTM(units=50, activation="relu", return_sequences=False), input_shape=input_shape),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'LSTM-Bidirectional', 'LSTM/'

    elif var == 15:  # LSTM com tamanhos de camada diferentes
        model = Sequential([
            LSTM(units=100, input_shape=input_shape, activation="relu", return_sequences=False),
            Dense(units=num_classes, activation=activation_output)
        ])
        model.compile(optimizer=optimizer, loss=loss)
        return model, 'LSTM-Large', 'LSTM/'

    else:
        raise ValueError("Parâmetro 'var' inválido. Deve estar entre 1 e 15.")


# Funções auxiliares
def create_directory(folder):
    os.makedirs(folder, exist_ok=True)

def train_model(model, X_train, Y_train, epochs, batch_size=32, val_split=0.1, patience=10):
    """
    Treina o modelo com Early Stopping.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=[early_stopping])
    return history

def plot_training_history(history, folder, model_name, mostrar_graficos=False):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda de treinamento', color='Orange')
    plt.plot(history.history['val_loss'], label='Perda de validação', color='Blue')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda de validação e treinamento ao longo do tempo')
    plt.legend()
    save_path = os.path.join(folder, f"{model_name}_training_history.png")
    plt.savefig(save_path)
    if mostrar_graficos:
        plt.show()
    plt.close()

def save_model(model, folder, model_name):
    save_path = os.path.join(folder, f"{model_name}.keras")
    model.save(save_path)

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

def plot_predictions_vs_expected(Y_true, Y_pred, folder, model_name, mostrar_graficos=False):
    """
    Plota e salva o gráfico de saídas previstas versus saídas esperadas.

    Parâmetros:
    - Y_true: Saídas esperadas (rótulos verdadeiros).
    - Y_pred: Saídas previstas pelo modelo.
    - folder: Diretório onde o gráfico será salvo.
    - model_name: Nome do modelo.
    - mostrar_graficos: Se True, exibe o gráfico.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(Y_true, label='Saídas Esperadas', color='Orange', alpha=0.6, linestyle='dashed')
    plt.plot(Y_pred, label='Saídas Previstas', color='Blue', alpha=0.6)
    plt.title('Saídas Previstas vs Saídas Esperadas')
    plt.xlabel('Índice das Amostras')
    plt.ylabel('Classe')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(folder, f"{model_name}_predictions_vs_expected.png")
    plt.savefig(save_path)
    if mostrar_graficos:
        plt.show()
    plt.close()
    print(f"Gráfico de saídas previstas x esperadas salvo em: {save_path}")

def evaluate_model(model, X, Y, folder, model_name, history, mostrar_graficos=False):
    """
    Avalia o modelo, salva os resultados e gera gráficos de saídas previstas versus esperadas.

    Parâmetros:
    - model: O modelo treinado.
    - X: Dados de entrada para avaliação.
    - Y: Rótulos verdadeiros.
    - folder: Diretório onde os resultados serão salvos.
    - model_name: Nome do modelo.
    - history: Histórico de treinamento (usado para obter o número ideal de épocas).
    - mostrar_graficos: Se True, exibe os gráficos gerados.
    """
    predictions = np.argmax(model.predict(X), axis=-1)
    acc = accuracy_score(Y, predictions)
    report = classification_report(Y, predictions)

    # Determinar o número ideal de épocas
    ideal_epochs = len(history.history['loss'])  # Número de épocas de treinamento efetivo

    # Imprimir métricas e número ideal de épocas
    print(f"Acurácia: {acc:.4f}\n{report}")
    print(f"Número ideal de épocas: {ideal_epochs}")

    # Salvar relatório e número ideal de épocas
    report_path = os.path.join(folder, f"{model_name}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Acurácia: {acc:.4f}\n\n{report}")
        f.write(f"\nNúmero ideal de épocas: {ideal_epochs}\n")

    # Gerar gráfico de saídas previstas versus esperadas
    plot_predictions_vs_expected(Y, predictions, folder, model_name, mostrar_graficos)

# Pipeline atualizado
epochs = [500]
patience = 10
model_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for var in model_types:
    for e in epochs:
        # Criar conjunto extra com a ordem original
        Y_extra = label_encoder.transform(saida)  # Transformar os rótulos originais para Y_extra
        X_extra = entrada.values.reshape((entrada.shape[0], 1, entrada.shape[1]))

        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(Y_train))
        model, model_name, folder = Modelos(var, input_shape, num_classes)
        create_directory(folder)
        history = train_model(model, X_train, Y_train, e, patience=patience)
        plot_training_history(history, folder, model_name)
        save_model(model, folder, model_name)
        evaluate_model(model, X_test, Y_test, folder, model_name, history, mostrar_graficos=False)
        aux = f"{model_name}_full"
        evaluate_model(model, X_extra, Y_extra, folder, aux, history, mostrar_graficos=False)

