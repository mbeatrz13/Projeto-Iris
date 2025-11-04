# som.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score



print("--- Análise com Mapa Auto-Organizável (SOM) ---")

# carregando o dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target # 0: Setosa, 1: Versicolor, 2: Virginica
target_names = iris.target_names

print("Dataset Iris carregado.")
print(f"Número de amostras: {X.shape[0]}, Número de características: {X.shape[1]}")
print("-" * 30)

# normaliza os dados para o intervalo [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# treinando o SOM

print("Iniciando o treinamento do SOM...")

#grid 3x3
grid_size = (3, 3)
input_len = X.shape[1] #  características dos dados de entrada (4)

# SOM 
som = MiniSom(x=grid_size[0], y=grid_size[1], input_len=input_len,
              sigma=0.5, learning_rate=0.5,
              random_seed=42)

# inicializa os pesos do SOM de forma aleatória
som.random_weights_init(X_scaled)

# treinamento do SOM
print(f"Treinando o SOM com uma grade {grid_size[0]}x{grid_size[1]}...")
som.train_random(data=X_scaled, num_iteration=1000)

print("Treinamento do SOM concluído.")
print("-" * 30)




# mapeia cada dado de entrada para o seu neurônio vencedor (BMU - Best Matching Unit)
#usamos um dicionário onde a chave é a coordenada do neurônio e o valor é uma lista de classes
winner_map = som.win_map(X_scaled)

# cria o mapa de rótulos: cada neurônio recebe o rótulo da classe mais frequente
label_map = np.full((grid_size[0], grid_size[1]), -1, dtype=int)
for position, samples in winner_map.items():
    if not samples:
        continue
    # pega os índices originais das amostras mapeadas para este neurônio
    sample_indices = [s.tolist() for s in samples]
    original_indices = [np.where((X_scaled == s).all(axis=1))[0][0] for s in sample_indices]
    
    # conta a classe mais comum entre essas amostras
    labels = y[original_indices]
    most_common = Counter(labels).most_common(1)
    if most_common:
        label_map[position] = most_common[0][0]

print("Mapa de rótulos do SOM (cada número representa uma classe):")
print(label_map)


# classificação e avaliação do SOM


y_pred_som = []
for x_s in X_scaled:
    winner_neuron = som.winner(x_s)
    pred_label = label_map[winner_neuron]
    y_pred_som.append(pred_label)

print("\nAvaliação da Classificação com SOM (nos dados de treino):")
print(f"Acurácia: {accuracy_score(y, y_pred_som):.4f}")
print("Relatório de Classificação SOM:")
print(classification_report(y, y_pred_som, target_names=target_names))
print("-" * 30)


# visualização

print("Gerando visualização do mapa com cores das classes...")
plt.figure(figsize=(10, 10)) 
plt.pcolor(som.distance_map().T, cmap='bone_r', edgecolors='k', linewidths=0.5) # Adiciona bordas aos neurônios
plt.colorbar(label='Distância Média dos Pesos dos Neurônios Vizinhos')

markers = ['o', 's', '^'] 
colors = ['red', 'green', 'blue'] # cores para cada classe (Setosa, Versicolor, Virginica)

# encontrando o neuronio vencedor 
for i, x in enumerate(X_scaled):
    w = som.winner(x) # w é a coordenada (linha, coluna) do neurônio vencedor
    
    # adiciona um pequeno ruído (jitter) para evitar sobreposição exata de pontos
    
    jit_x = w[0] + 0.5 + (np.random.rand() - 0.5) * 0.8 
    jit_y = w[1] + 0.5 + (np.random.rand() - 0.5) * 0.8 
    
    #plota o marcador com a cor e forma 
    plt.plot(jit_x, jit_y,
             markers[y[i]], 
             markeredgecolor=colors[y[i]],
             markerfacecolor=colors[y[i]], 
             markersize=10, 
             markeredgewidth=1) 

#rotula cada neurônio 
for r in range(grid_size[0]):
    for c in range(grid_size[1]):
        if label_map[r, c] != -1: # se o neurônio foi mapeado para alguma classe
            plt.text(r + 0.5, c + 0.5, target_names[label_map[r, c]],
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='k')) 

plt.title('Mapa Auto-Organizável (SOM) para o Dataset Iris')
plt.xticks(np.arange(grid_size[0] + 1), labels=[f'Col {i}' for i in range(grid_size[0] + 1)])
plt.yticks(np.arange(grid_size[1] + 1), labels=[f'Row {i}' for i in range(grid_size[1] + 1)])
plt.xlabel('Dimensão X do Mapa')
plt.ylabel('Dimensão Y do Mapa')
plt.grid(True, linestyle='--', alpha=0.6)

# legenda
legend_elements = [
    plt.Line2D([0], [0], marker=markers[0], color='w', label=target_names[0],
               markerfacecolor=colors[0], markersize=10, markeredgecolor=colors[0]),
    plt.Line2D([0], [0], marker=markers[1], color='w', label=target_names[1],
               markerfacecolor=colors[1], markersize=10, markeredgecolor=colors[1]),
    plt.Line2D([0], [0], marker=markers[2], color='w', label=target_names[2],
               markerfacecolor=colors[2], markersize=10, markeredgecolor=colors[2])
]
plt.legend(handles=legend_elements, loc='upper left', title='Classes Verdadeiras')

plt.show()

