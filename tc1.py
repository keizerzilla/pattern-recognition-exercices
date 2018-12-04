"""
tc1.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
NOV/2018

==========================
Trabalho Computacional #01
==========================

Sobre: classificadores lineares e quadráticos, métricas de desempenho e redução
de dimennsionalidade
Requisitos: recpad.py
"""

from recpad import *

classifiers = {"NN"  : NN(n_neighbors=1),
               "DMC" : DMC(),
               "CQG" : CQG(store_covariance=True)}
dataset = "data/parkinsons.data"
test_rate = 0.3
rounds = 100

df = pd.read_csv(dataset)
df = df.drop(["name"], axis=1)
X = df.drop(["status"], axis=1)
y = df["status"]

ans = classify(classifiers, X, y, test_rate, 1)
sumary(ans, "analisando matriz de confusao")

"""
ans = classify(classifiers, X, y, test_rate, rounds)
sumary(ans, "Desempenho dos classificadores sem reducao de dimensionalidade")
find_best_pca(dataset, classifiers, test_rate, "figures/precisao-pca.png")
X_lda = reduction_lda(X, y, 1)
ans = classify(classifiers, X_lda, y, test_rate, rounds)
sumary(ans, "Desempenho dos classificadores com reducao usando LDA")
"""

X_data = np.array(X)
covm = np.cov(X_data.T)
rank_covm = np.linalg.matrix_rank(covm)
cond_covm = np.linalg.cond(covm)
print("Shape da matriz de covariancia: {}".format(covm.shape))
print("Posto da matriz de covariancia: {}".format(rank_covm))
print("Cond da matrix de covariancia: {}".format(cond_covm))
np.linalg.inv(np.array(covm))






