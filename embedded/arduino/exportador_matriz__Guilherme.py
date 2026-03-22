import openpyxl
import pandas as pd

file_name = "dados_normalizados_s.xlsx"
raw_data = pd.read_excel(file_name, usecols="B,C,D,E,F",names=['Tempo','Externo','Saida','Interno','Porta'])

matriz = raw_data.values.tolist()

colunas = len(matriz[0])
linhas = len(matriz)

parte1 = matriz[:3000]
parte2 = matriz[3000:]


matriz1 = []
matriz2 = []
matriz3 = []
matriz4 = []
matriz5 = []
for i in range(linhas):
    if i < 2001: 
        matriz1.append(matriz[i])
    if i>= 2001 and i< 3001:
        matriz2.append(matriz[i])
    if i>= 3001 and i< 4001:
        matriz3.append(matriz[i])
    if i>=4001 and i<6001:
        matriz4.append(matriz[i])
    if i>=6001 and i<linhas:
        matriz5.append(matriz[i])

# Abra um arquivo de saída em C
with open('dados_normalizados__Guilherme_p1.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in matriz1:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")

with open('dados_normalizados__Guilherme_p2.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in matriz2:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")

with open('dados_normalizados__Guilherme_p3.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in matriz3:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")

with open('dados_normalizados__Guilherme_p4.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in matriz4:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")


with open('dados_normalizados__Guilherme_p5.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in matriz5:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")

with open('dados_normalizados__Guilherme.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in matriz:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")

with open('dados_normalizados__Guilherme_parte1.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in parte1:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")

with open('dados_normalizados__Guilherme_parte2.h', 'w') as f:
    f.write("#ifndef DADOS_EM_C_H\n")
    f.write("#define DADOS_EM_C_H\n\n")
    f.write(f"const int COLUNAS = {colunas};\n")
    # Escreva os dados em C
    f.write("float dados[][COLUNAS] = {\n")
    for linha in parte2:
        f.write("    {")
        f.write(", ".join(str(valor) for valor in linha))
        f.write("},\n")
    f.write("};\n\n")

    f.write("#endif\n")