import os

def combinar_arquivos_txt(pasta_origem, arquivo_saida):
    try:
        # Lista todos os arquivos na pasta de origem
        arquivos = [f for f in os.listdir(pasta_origem) if f.endswith('.txt')]
        
        # Abre o arquivo de saída no modo de escrita
        with open(arquivo_saida, 'w', encoding='utf-8') as arquivo_final:
            for arquivo in arquivos:
                caminho_arquivo = os.path.join(pasta_origem, arquivo)
                try:
                    # Tenta abrir o arquivo em UTF-8, e se falhar, tenta outra codificação
                    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo_txt:
                        conteudo = arquivo_txt.read()
                except UnicodeDecodeError:
                    # Se UTF-8 falhar, tenta codificação alternativa
                    with open(caminho_arquivo, 'r', encoding='latin1') as arquivo_txt:
                        conteudo = arquivo_txt.read()
                
                # Escreve o conteúdo no arquivo final com uma separação
                arquivo_final.write(f"--- Início do arquivo: {arquivo} ---\n")
                arquivo_final.write(conteudo)
                arquivo_final.write(f"\n--- Fim do arquivo: {arquivo} ---\n\n")
        
        print(f"Arquivos combinados com sucesso em '{arquivo_saida}'!")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Defina o caminho da pasta contendo os arquivos .txt e o nome do arquivo de saída
pasta = r"pasta"  # Altere para o caminho da sua pasta
arquivo_resultado = os.path.join(pasta, 'resultado.txt')

combinar_arquivos_txt(pasta, arquivo_resultado)
