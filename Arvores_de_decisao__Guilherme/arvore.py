import csv
import sklearn
import matplotlib 
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import emlearn
from micromlgen import port

plot_v = True

#definindo a profundidade da arvore
depth = 15
#coletando e separando os dados
file_name = "dados_normalizados_s.xlsx"
raw_data = pd.read_excel(file_name, usecols="B,C,D,E,F",names=['Tempo','Externo','Saida','Interno','Porta'])

#file_name = "dados_normalizados_s.xlsx"
#raw_data = pd.read_excel(file_name, usecols="B,C,D,E,F",names=['Tempo','Externo','Saida','Interno','Porta'])

data = raw_data.copy()
data = data.apply (pd.to_numeric, errors='coerce')
data = data.dropna()
print ("Number of nan:", data.isnull().sum().sum())

test_in = data.copy()
test_out = test_in.pop('Saida')


train_in = test_in.copy()
train_out = test_out.copy()

#construindo e treinando a arvore
for depth in range(21,101,5):
    train_in = test_in.copy()

    tree_reg = DecisionTreeRegressor(max_depth=depth)
    tree_reg.fit(train_in, train_out)
    test_pred = tree_reg.predict(train_in)

    cmodel = emlearn.convert(tree_reg, method='inline')
    nome_arquivo = 'Arvores_de_decisao__Guilherme'
    nome_arquivo = nome_arquivo + str(depth) + '.h'
    cmodel.save(file=nome_arquivo, name='Arvores_de_decisao__Guilherme')

    nome_arquivo2 = 'microMLgen-Arvores_de_decisao__Guilherme'
    nome_arquivo2 = nome_arquivo2 + str(depth) + '.h'
    c_code = port(tree_reg)
    with open(nome_arquivo2, 'w') as file:
        file.write(c_code)

    # calculando o erro medio quadratico
    mse = mean_squared_error(train_out, test_pred)

    # escrevendo dados de erro em um arquivo csv
    with open('mse_norm_tree.csv', 'a', encoding="UTF8", newline='') as file:
        writer = csv.writer(file)
        mse_data = [mse, depth, "-"]
        writer.writerow(mse_data)
        print(mse_data)

    #plotando o grafico de predicao x valores coletados
    if(plot_v == True):
        plt.figure(figsize=(19.25, 10.23))
        plt.plot(list(range(0, len(test_out))),test_pred)
        plt.plot(list(range(0, len(test_out))),test_out,linestyle = ':')
        #plt.savefig("tree_depth_"+ str(depth) + "_mse_"+ str(round(mse, 4)) +".pdf")
        plt.savefig("tree_depth_"+ str(depth) + "_mse_"+ str(round(mse, 4)) +".png")
        plt.close()
