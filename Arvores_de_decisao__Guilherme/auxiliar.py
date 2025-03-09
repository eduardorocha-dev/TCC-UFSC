
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from sklearn.metrics import mean_squared_error
from math import sqrt





vetor = []

nome_arquivo = 'dados_microMLgen.txt'

# Carregar os números do arquivo para um vetor usando numpy
vetor = np.loadtxt(nome_arquivo)

print(vetor)


file_name = "dados_normalizados_s.xlsx"
raw_data = pd.read_excel(file_name, usecols="B,C,D,E,F",names=['Tempo','Externo','Saida','Interno','Porta'])


data = raw_data.copy()

test_in = data.copy()
test_out = test_in.pop('Saida')

plt.figure(figsize=(19.25, 10.23))
plt.plot(list(range(0, len(vetor))),vetor)
plt.plot(list(range(0, len(test_out))),test_out,linestyle = ':')
plt.savefig("Imagem_microMLgen.png")
plt.close()

saida_coluna = raw_data['Saida']
vetor_numpy_saida = saida_coluna.values
print(vetor_numpy_saida)

valores_reais = vetor_numpy_saida
valores_previstos = vetor

indices_nan = np.isnan(valores_reais)
valores_reais_sem_nan = valores_reais[~indices_nan]