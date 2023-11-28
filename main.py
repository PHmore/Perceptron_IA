# Carregar o conjunto de dados (substitua 'seu_arquivo.csv' pelo seu arquivo de dados)
#data = pd.read_csv('seu_arquivo.csv')

import pandas as pd
from sklearn.datasets import load_iris  # Importar o conjunto de dados Iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregar o conjunto de dados Iris do sklearn
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Separar as variáveis de entrada (X) e a variável de saída (y)
X = data.drop('target', axis=1)  # Features
y = data['target']  # Variável de destino

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar e treinar o modelo MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Fazer previsões
y_pred = mlp.predict(X_test)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# Imprimir a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:\n", conf_matrix)


"""
    Importação das bibliotecas necessárias:
        pandas para manipulação de dados.
        load_iris do sklearn.datasets para carregar o conjunto de dados Iris.
        train_test_split para dividir os dados em conjuntos de treino e teste.
        MLPClassifier para criar o classificador de redes neurais.
        accuracy_score e confusion_matrix para avaliar o desempenho do modelo.

    Carregamento e preparação do conjunto de dados Iris:
        Usamos load_iris() para carregar os dados.
        Convertendo os dados para um DataFrame do Pandas para facilitar a manipulação.
        iris.feature_names contém os nomes das características (features).
        iris.target contém os rótulos de classe.

    Divisão dos dados em conjuntos de treino e teste:
        Utilizamos train_test_split para separar os dados em 80% para treino e 20% para teste.

    Configuração e treinamento do modelo MLPClassifier:
        Criamos uma instância do classificador MLPClassifier com duas camadas escondidas de tamanhos 100 e 50, respectivamente.
        Definimos o número máximo de iterações como 500 e o estado aleatório para garantir a reprodutibilidade.

    Previsões e avaliação do modelo:
        Utilizamos o modelo treinado para fazer previsões nos dados de teste.
        Calculamos a acurácia do modelo usando accuracy_score.
        Exibimos a matriz de confusão para avaliar o desempenho em cada classe usando confusion_matrix.
"""