# svm.py

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print("--- Classificação com Máquina de Vetores de Suporte (SVM) ---")

# carrega o dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("Dataset Iris carregado.")


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDados divididos em {len(X_train)} amostras de treino e {len(X_test)} de teste.")
print("-" * 30)


# treinando a SVM

print("Iniciando o treinamento do SVM com kernel RBF...")

# Inicialização do SVM com kernel RBF
# 'gamma'='auto' usa 1 / n_features
# 'C'=1.0 é o parâmetro de regularização padrão
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto', random_state=42)

# treinamento do SVM com os dados de treino
svm_classifier.fit(X_train, y_train)

print("Treinamento do SVM concluído.")
print("-" * 30)


#avaliacao do modelo

print("Avaliando o modelo no conjunto de teste...")

# predição nos dados de teste
y_pred_svm = svm_classifier.predict(X_test)

# metricas de avaliação
print("\nResultados da Avaliação do SVM:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_svm):.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_svm))

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred_svm, target_names=target_names))
print("-" * 30)