import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Dados AND
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([-1, -1, -1, 1])  # -1 e +1

# Treina SVM linear
svm = SVC(kernel='linear', C=1e5)
svm.fit(X, y)

# Coeficientes do hiperplano
w = svm.coef_[0]
b = svm.intercept_[0]

print(f"w = {w}")
print(f"b = {b}")

# Visualização
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, cmap='plasma', s=100)

# Gera reta separadora
x1 = np.linspace(-0.5, 1.5, 10)
x2 = -(w[0]*x1 + b)/w[1]
plt.plot(x1, x2, 'k--')

plt.title("SVM Linear - Problema AND")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True)
plt.show()
