import serial
import matplotlib.pyplot as plt
import pandas as pd

# Configurar a porta serial e a taxa de transmissão (baud rate)
arduino_port = '/dev/ttyUSB0'  # Substitua pela porta serial correta
baud_rate = 115200

# Inicializar a conexão serial
ser = serial.Serial(arduino_port, baud_rate)

# Nome do arquivo onde os dados serão salvos
file_name = "raw-arduino-dados_arvore_de_descisao__Guilherme_1.txt"

valores = []
# Abrir o arquivo para escrita
with open(file_name, "w") as file:
    while len(valores) < (3000):
        data = ser.readline().decode('utf-8').strip()  # Ler uma linha da porta serial
        if data:  # Verifica se há dados
            print(data)  # Exibe os dados no console (opcional)
            file.write(data + "\n")  # Escreve os dados no arquivo de texto
            valores.append(data)

# Fechar a conexão serial
ser.close()
#4575-6