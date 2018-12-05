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

#====================================
# Dados e variáveis gerais do script
#====================================

sb.set()

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

X0 = df.loc[df["status"] == 0]
X0 = X0.drop(["status"], axis=1)
X1 = df.loc[df["status"] == 1]
X1 = X1.drop(["status"], axis=1)

#================================================
# Analise dos postos das matrizes de covariancia
#================================================

print("Posto matriz de cov C0: {}".format(rank_covmatrix(X0)))
print("Posto matriz de cov C1: {}".format(rank_covmatrix(X1)))

X0_norm = super_normalize(X0)
X1_norm = super_normalize(X1)

print("Posto matriz de cov C0 norm: {}".format(rank_covmatrix(X0_norm)))
print("Posto matriz de cov C1 norm: {}".format(rank_covmatrix(X1_norm)))

X0_unskew = super_normalize(X0)
X1_unskew = super_normalize(X1)

print("Posto matriz de cov C0 unskew: {}".format(rank_covmatrix(X0_unskew)))
print("Posto matriz de cov C1 unskew: {}".format(rank_covmatrix(X1_unskew)))

#==============================================
# Desempenho geral, PCA e LDA sem normalizacao
#==============================================

ans = classify(classifiers, X, y, test_rate, rounds)
sumary(ans, "Desempenho geral")

find_best_pca(dataset, classifiers, test_rate,
              "figures/tc1/tc1-precisao-pca-geral.png")

X_lda = reduction_lda(X, y, 1)
ans = classify(classifiers, X_lda, y, test_rate, rounds)
sumary(ans, "Desempenho LDA")

#==============================================
# Desempenho geral, PCA e LDA com normalizacao
#==============================================

ans = classify(classifiers, X, y, test_rate, rounds, normalize=True)
sumary(ans, "Desempenho geral (normalizado)")

find_best_pca(dataset, classifiers, test_rate,
              "figures/tc1/tc1-precisao-pca-normalizado.png", normalize=True)

X_lda = reduction_lda(X, y, 1)
ans = classify(classifiers, X_lda, y, test_rate, rounds, normalize=True)
sumary(ans, "Desempenho LDA (normalizado)")

#======================================================================
# Desempenho geral, PCA e LDA com normalizacao e remocao de assimetria
#======================================================================

ans = classify(classifiers, X, y, test_rate, rounds, normalize=True,
               unskew=True)
sumary(ans, "Desempenho geral (normalizado, unskew)")

find_best_pca(dataset, classifiers, test_rate,
              "figures/tc1/tc1-precisao-pca-normalizado-unskew.png",
              normalize=True, unskew=True)

X_lda = reduction_lda(X, y, 1)
ans = classify(classifiers, X_lda, y, test_rate, rounds, normalize=True,
               unskew=True)
sumary(ans, "Desempenho LDA (normalizado, unskew)")

