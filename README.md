# Aplicações de Inteligência Artificial em Dispositivos Embarcados para Monitoramento Térmico de Vacinas  

Este repositório contém o código e os materiais desenvolvidos para o meu Trabalho de Conclusão de Curso (TCC) em Engenharia Mecatrônica na Universidade Federal de Santa Catarina (UFSC) – Centro Tecnológico de Joinville.  

O projeto investiga a implementação de sistemas de inteligência artificial (IA) em microcontroladores ESP32, com foco na adaptação e otimização de modelos previamente desenvolvidos em ambiente computacional para um contexto de hardware embarcado. O principal desafio abordado foi a transferência dessas funcionalidades para dispositivos com restrições de memória e capacidade de processamento, visando aplicações na gestão da cadeia de frio para vacinas.  

## 📌 Objetivos  
- Implementar modelos de aprendizado de máquina em microcontroladores para predição do estado da porta e da temperatura interna de refrigeradores usados no armazenamento de vacinas.  
- Comparar o desempenho dos modelos implementados no microcontrolador com os processados em computador.  
- Desenvolver modelos de redes neurais artificiais para aprimorar a eficiência dos sistemas embarcados.  
- Avaliar o impacto da importação dos modelos para um ambiente de hardware com recursos limitados.  

## 🛠️ Tecnologias Utilizadas  
- **Microcontrolador ESP32**  
- **Linguagens e Bibliotecas:**  
  - Python (TensorFlow, scikit-learn, NumPy)  
  - C (para integração com microcontroladores)  
  - TinyML e emlearn (para conversão e otimização de modelos)  
- **Modelos de IA:**  
  - Árvores de Decisão  
  - Redes Neurais Recorrentes (SimpleRNN, GRU, LSTM)  

## 📊 Resultados  
Os resultados demonstraram que, apesar de uma leve redução no desempenho dos modelos após a conversão para ambiente embarcado, os sistemas se mostraram eficientes na tarefa de monitoramento térmico. A implementação de redes neurais possibilitou melhorias na predição do estado da porta e da temperatura do buffer térmico, contribuindo para a confiabilidade da solução proposta.  

## 📜 Contribuição  
Este projeto contribui para o avanço de soluções inteligentes voltadas para dispositivos de baixo consumo energético, promovendo inovações na Internet das Coisas (IoT) e na preservação de vacinas em locais com infraestrutura limitada.  
