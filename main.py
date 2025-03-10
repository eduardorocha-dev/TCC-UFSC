import os
import matplotlib.pyplot as plt
import pandas as pd
from modelos import Modelos
from Git.dados import X_train, X_test, Y_train, Y_test, entrada, saida
from tensorflow.keras.callbacks import EarlyStopping  # Importa EarlyStopping

from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def create_directory(folder):
    """
    Cria um diretório se ele não existir.
    """
    os.makedirs(folder, exist_ok=True)

def save_training_epochs(folder, model_name, total_epochs, stopped_epoch):
    """
    Salva as informações das épocas de treinamento em um arquivo .txt.
    """
    save_path = os.path.join(folder, f"{model_name}_epochs.txt")
    with open(save_path, 'w') as file:
        file.write(f"Total de épocas configuradas: {total_epochs}\n")
        if stopped_epoch > 0:
            file.write(f"Treinamento interrompido antecipadamente na época: {stopped_epoch + 1}\n")
        else:
            file.write("Treinamento completado sem interrupção antecipada.\n")
    print(f"Informações das épocas salvas em: {save_path}")

def train_model(model, X_train, Y_train, epochs, batch_size=32, val_split=0.1, folder=None, model_name=None):
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitora a perda na validação
        patience=10,          # Espera 5 épocas sem melhora antes de parar
        restore_best_weights=True  # Restaura os melhores pesos
    )

    history = model.fit(
        X_train, Y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=val_split, 
        callbacks=[early_stopping]  # Adiciona o EarlyStopping como callback
    )
    
    # Calcula a época onde o treinamento parou devido ao EarlyStopping
    stopped_epoch = early_stopping.stopped_epoch
    save_training_epochs(folder, model_name, epochs, stopped_epoch)
    
    return history

def plot_training_history(history, folder, model_name, mostrar_graficos=False):
    """
    Plota a curva de treinamento e validação ao longo do tempo e salva o gráfico em escala de cinza.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda de treinamento', color='Red')
    plt.plot(history.history['val_loss'], label='Perda de validação', color='Blue')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Perda de validação e treinamento ao longo do tempo')
    plt.legend()
    
    save_path = os.path.join(folder, f"{model_name}_training_history.png")
    plt.savefig(save_path)
    
    if mostrar_graficos:
        plt.show()
    
    print(f"Gráfico de histórico de treinamento salvo em: {save_path}")
    plt.close()

def save_model(model, folder, model_name):
    """
    Salva o modelo treinado no formato .keras.
    """
    save_path = os.path.join(folder, f"{model_name}.keras")
    model.save(save_path)
    print(f"Modelo salvo em: {save_path}")

def plot_RNN(x, y, folder, model_name, loss, mae, r2, mostrar_graficos=False):
    """
    Plota as saídas reais vs previstas e salva o gráfico em escala de cinza.
    """
    
    save_text_path = os.path.join(folder, f"{model_name}_data.txt")
    with open(save_text_path, 'w') as arquivo:
        texto = f"Test Loss: {loss}\nMean Absolute Error (MAE): {mae}\nR² Score: {r2}"
        arquivo.write(texto)
    
    save_plot_path = os.path.join(folder, f"{model_name}_plot.png")
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='Dados alvo', linestyle='--', color='Red')
    plt.plot(x, label='Dados previstos', color='Blue')
    plt.title('Saídas esperadas e previstas')
    plt.xlabel('Amostras')
    plt.ylabel('Saída')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_plot_path)
    
    if mostrar_graficos:
        plt.show()
    
    print(f"Gráfico de resultados salvo em: {save_plot_path}")
    plt.close()

def evaluate_model(model, X, Y, predictions, folder, model_name, mostrar_graficos=False):
    """
    Avalia o modelo e plota os resultados.
    """
    loss = model.evaluate(X, Y, verbose=0)
    mae = mean_absolute_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    
    plot_RNN(predictions, Y, folder, model_name, loss, mae, r2, mostrar_graficos)

def run_model_pipeline(model_func, epochs, var, mostrar_graficos=False):
    """
    Configura, treina, avalia e salva o modelo.
    """
    global X_train, X_test, Y_train, Y_test, entrada, saida

    # Configura o modelo e diretório
    model, model_name, folder = model_func(var)
    create_directory(folder)

    # Compila o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treina o modelo e obtém o histórico de treinamento
    history = train_model(model, X_train, Y_train, epochs, folder=folder, model_name=model_name)

    # Plota e salva o histórico de treinamento
    model_name_with_epochs = f"{epochs}-{model_name}"
    plot_training_history(history, folder, model_name_with_epochs, mostrar_graficos)

    # Salva o modelo
    save_model(model, folder, model_name_with_epochs)

    # Avaliação no conjunto de teste
    predictions_test = model.predict(X_test)
    evaluate_model(model, X_test, Y_test, predictions_test, folder, model_name_with_epochs, mostrar_graficos)

    # Avaliação em toda a amostra
    predictions_full = model.predict(entrada)
    model_name_full = f"{model_name_with_epochs}_full"
    evaluate_model(model, entrada, saida, predictions_full, folder, model_name_full, mostrar_graficos)

# Parâmetros para execução do pipeline
epocas = [500]
model_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
mostrar_graficos = False  # Defina como True para mostrar gráficos na tela

# Executa o pipeline para diferentes tipos de modelos e épocas
for var in model_types:
    for epochs in epocas:
        run_model_pipeline(Modelos, epochs, var, mostrar_graficos)
