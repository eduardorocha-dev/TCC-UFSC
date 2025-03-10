import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from Git.dados import X_train, X_test, Y_train, Y_test, entrada, saida

def load_keras_model(filepath):
    """
    Carrega um modelo salvo no formato .keras.
    """
    try:
        model = load_model(filepath)
        print(f"Modelo carregado de: {filepath}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def evaluate_model(model, X, Y):
    """
    Avalia o modelo com os dados fornecidos e calcula métricas.
    """
    predictions = model.predict(X)
    loss = model.evaluate(X, Y, verbose=0)
    mae = mean_absolute_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    return loss, mae, r2

def collect_and_evaluate_models(directory, output_dir):
    """
    Coleta todos os arquivos .keras em um diretório, avalia os modelos e salva os resultados.
    """
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".keras"):
            filepath = os.path.join(directory, filename)
            model = load_keras_model(filepath)
            if model:
                # Avalia no conjunto de teste
                loss_test, mae_test, r2_test = evaluate_model(model, X_test, Y_test)
                
                # Avalia no conjunto completo (entrada)
                loss_full, mae_full, r2_full = evaluate_model(model, entrada, saida)
                
                results.append({
                    "model_name": filename,
                    "loss_test": loss_test,
                    "mae_test": mae_test,
                    "r2_test": r2_test,
                    "loss_full": loss_full,
                    "mae_full": mae_full,
                    "r2_full": r2_full,
                })
    return results

def save_metrics_to_file(results, output_dir):
    """
    Salva as métricas de avaliação em um arquivo de texto.
    """
    save_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(save_path, "w") as f:
        for result in results:
            f.write(f"Modelo: {result['model_name']}\n")
            f.write(f"Perda no Teste: {result['loss_test']}\n")
            f.write(f"MAE no Teste: {result['mae_test']}\n")
            f.write(f"R² no Teste: {result['r2_test']}\n")
            f.write(f"Perda no Conjunto Completo: {result['loss_full']}\n")
            f.write(f"MAE no Conjunto Completo: {result['mae_full']}\n")
            f.write(f"R² no Conjunto Completo: {result['r2_full']}\n")
            f.write("\n")
    print(f"Resumo salvo em: {save_path}")

def plot_metrics(results, output_dir):
    """
    Plota as métricas de perda, MAE e R² para os modelos avaliados.
    """
    # Cria listas para cada métrica
    model_names = [result["model_name"] for result in results]
    loss_test = [result["loss_test"] for result in results]
    mae_test = [result["mae_test"] for result in results]
    r2_test = [result["r2_test"] for result in results]
    loss_full = [result["loss_full"] for result in results]
    mae_full = [result["mae_full"] for result in results]
    r2_full = [result["r2_full"] for result in results]

    # Cria os gráficos
    metrics = [
        ("Perda de validação", loss_test),
        ("MAE de validação", mae_test),
        ("R² de validação", r2_test),
        ("Perda no Conjunto Completo", loss_full),
        ("MAE no Conjunto Completo", mae_full),
        ("R² no Conjunto Completo", r2_full),
    ]

    for metric_name, metric_values in metrics:
        plt.figure(figsize=(10, 6))
        
        bar_width = 0.5  # Ajuste da largura das barras
        bars = plt.bar(model_names, metric_values, color='blue', width=bar_width)
        
        # Adicionar os valores no topo de cada barra
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # Posição X: centro da barra
                height,  # Posição Y: topo da barra
                f'{height:.2f}',  # Texto formatado
                ha='center', va='bottom', fontsize=10, color='black'  # Ajuste da aparência do texto
            )
        
        plt.xticks(rotation=45, ha="right")
        plt.title(metric_name)
        plt.ylabel(metric_name)
        plt.xlabel("Modelos")
        plt.tight_layout()
        
        # Salva o gráfico
        save_path = os.path.join(output_dir, f"{metric_name.replace(' ', '_')}.png")
        plt.savefig(save_path)
        print(f"Gráfico salvo em: {save_path}")
        plt.close()

def create_directory(folder):
    """
    Cria um diretório se ele não existir.
    """
    os.makedirs(folder, exist_ok=True)

# Exemplo de uso
if __name__ == "__main__":
    # Diretórios das pastas de modelos
    base_dir = ""  # Substitua pelo caminho das pastas GRU, LSTM e SRNN
    output_base_dir = "_metricas"  # Diretório base para os resultados
    create_directory(output_base_dir)

    # Pastas a serem processadas
    subfolders = ["GRU", "LSTM", "SRNN"]

    for subfolder in subfolders:
        model_dir = os.path.join(base_dir, subfolder)
        output_dir = os.path.join(output_base_dir, subfolder)
        create_directory(output_dir)

        print(f"\nProcessando a pasta: {subfolder}")
        
        # Avalia os modelos na pasta
        results = collect_and_evaluate_models(model_dir, output_dir)

        # Salva as métricas em arquivo e gera gráficos
        save_metrics_to_file(results, output_dir)
        plot_metrics(results, output_dir)
