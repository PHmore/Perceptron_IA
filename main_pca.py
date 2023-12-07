import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Carregar o conjunto de dados Iris
iris_data = load_iris()

# Separar os dados em variáveis de entrada (features) e saída (target)
X = iris_data.data  # Características das flores (4 características)
y = iris_data.target  # Classes das flores (0, 1, 2 para setosa, versicolor, virgínica)

# Redução de dimensionalidade com PCA para 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste (70% treinamento, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Inicializar o modelo MLPClassifier
model = MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(4,), alpha=0.001, solver='lbfgs')

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões usando os dados de teste
predictions = model.predict(X_test)

# Avaliar o modelo
cm = metrics.confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(cm)

accuracy = metrics.accuracy_score(y_test, predictions)
print("Acurácia:", accuracy)

# Plot dos pontos de entrada com PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, edgecolors='k', label='Classes')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização das classes do conjunto Iris com PCA')
plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.show()
